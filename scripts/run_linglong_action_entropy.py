"""Per-frame conditional EEF action entropy for Linglong fittings demos.

Each sampled frame is denoised with several action noises. Predictions of the
same timestamp from overlapping chunks are pooled, and the entropy is the
Gaussian entropy of both-hand xyz. Global ranks split episodes by sampled-frame
count and write one npz per episode. torchrun may span several machines; the
script does not initialize a process group.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.deploy.denoise_schedule import make_schedule
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.deploy.server import _load_deploy_yaml, merge_deploy_cfg

XYZ = np.array([0, 1, 2, 8, 9, 10], dtype=np.int64)
GRIP = np.array([7, 15], dtype=np.int64)
DEFAULT_CKPT = (
    "/vePFS/tangyuting/OpenWAM/outputs/"
    "linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu_bs4_official_finetune/"
    "2026-10-19_15-10-08"
)
DEFAULT_CKPT_NAME = "checkpoint_step_18185.safetensors"
DEFAULT_DATA = (
    "/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/"
    "data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_train_15hz"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CKPT)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CKPT_NAME)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--noises", type=int, default=8)
    parser.add_argument("--group", type=int, default=3, help="Observation cases packed into one forward, each with --noises.")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--episodes", default="", help="Comma-separated episode indices. Default: every train episode.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _rank_world() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world < 1 or rank < 0 or rank >= world or local_rank < 0:
        raise ValueError(f"invalid rank={rank} local_rank={local_rank} world={world}")
    return rank, local_rank, world


def _device(local_rank: int) -> str:
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) == 1:
        return "cuda:0"
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"local_rank={local_rank} but only {torch.cuda.device_count()} GPUs are visible")
    return f"cuda:{local_rank}"


def _repeat_batch(tensor, batch):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] != 1:
        return tensor
    return tensor.repeat(batch, *([1] * (tensor.ndim - 1)))


def _cat_pad(values, dim):
    if dim != 1:
        return torch.cat(values, dim=0)
    max_len = max(value.shape[1] for value in values)
    padded = []
    for value in values:
        if value.shape[1] == max_len:
            padded.append(value)
            continue
        pad_shape = list(value.shape)
        pad_shape[1] = max_len - value.shape[1]
        fill = False if value.dtype == torch.bool else 0
        padded.append(torch.cat([value, value.new_full(pad_shape, fill)], dim=1))
    return torch.cat(padded, dim=0)


@torch.no_grad()
def _denoise(arch, inputs, action_latents, schedule):
    inactive = arch._resolve_inactive_action_dims(None, arch.device)
    inactive_noise = None
    ref = inputs.get("first_frame_latents")
    batch = action_latents.shape[0]
    num_train_ts_v = float(arch.video_scheduler.num_train_timesteps)
    num_train_ts_a = float(arch.action_scheduler.num_train_timesteps)
    dtype = arch.dtype
    device = arch.device
    for i in range(len(schedule) - 1):
        t_v, t_a = schedule[i]
        t_v_next, t_a_next = schedule[i + 1]
        sigma_v = t_v / num_train_ts_v
        sigma_a = t_a / num_train_ts_a
        sigma_v_next = t_v_next / num_train_ts_v
        sigma_a_next = t_a_next / num_train_ts_a
        if sigma_v == sigma_v_next and sigma_a == sigma_a_next:
            continue
        v_timestep = torch.full((batch,), float(t_v), dtype=dtype, device=device)
        a_timestep = torch.full((batch,), float(t_a), dtype=dtype, device=device)
        noise_pred, action_noise_pred = arch.forward(
            action_latents, a_timestep, **inputs, timestep=v_timestep
        )
        if sigma_v != sigma_v_next:
            new_latents = inputs["latents"] + noise_pred * (sigma_v_next - sigma_v)
            if ref is not None:
                new_latents = new_latents.clone()
                new_latents[:, :, : ref.shape[2]] = ref
            inputs["latents"] = new_latents
        if sigma_a != sigma_a_next and action_noise_pred is not None:
            if inactive is not None and inactive_noise is None:
                inactive_noise = action_latents[..., inactive].detach().clone() / float(sigma_a)
            action_latents = arch.action_scheduler.flow_step(
                action_noise_pred, sigma_a, sigma_a_next, action_latents
            )
            if inactive is not None:
                action_latents[..., inactive] = inactive_noise * float(sigma_a_next)
    return action_latents


def _preprocess_case(engine, image, proprio, prompt, copies, height, width):
    arch = engine.architecture
    inf = engine.cfg.inference
    shift = getattr(arch.action_backbone, "shift_action", None) or 5.0
    inputs = arch.video_backbone.preprocess_input_for_inference(
        prompt=prompt,
        first_frame_image=[image],
        num_frames=int(getattr(inf, "video_num_frames", 9)),
        height=height,
        width=width,
        seed=0,
        num_inference_steps=int(getattr(inf, "denoise_steps", 10)),
        shift=shift,
        tiled=True,
        prompt_embed_cache=engine._prompt_embed_cache,
    )
    proprio_t = arch.normalize_deploy_proprio(proprio).to(device=arch.device, dtype=arch.dtype)
    if proprio_t.ndim == 1:
        proprio_t = proprio_t.unsqueeze(0)
    inputs["proprio"] = proprio_t
    for key in (
        "latents",
        "input_latents",
        "context",
        "proprio",
        "first_frame_latents",
        "seq_lens",
        "context_mask",
        "condition_mask",
    ):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            inputs[key] = _repeat_batch(inputs[key], copies)
    ref = inputs.get("first_frame_latents")
    latents = torch.randn_like(inputs["latents"])
    if ref is not None:
        latents[:, :, : ref.shape[2]] = ref
    inputs["latents"] = latents
    return inputs


def _stack_cases(cases):
    keys = (
        "latents",
        "input_latents",
        "context",
        "proprio",
        "first_frame_latents",
        "seq_lens",
        "context_mask",
        "condition_mask",
    )
    stacked = dict(cases[0])
    for key in keys:
        values = [case[key] for case in cases if key in case and isinstance(case[key], torch.Tensor)]
        if len(values) != len(cases):
            continue
        stacked[key] = _cat_pad(values, 1 if key == "context" else 0)
    return stacked


def _make_schedule(engine, steps: int):
    arch = engine.architecture
    inf = engine.cfg.inference
    return make_schedule(
        "sync",
        video_scheduler=arch.video_scheduler,
        action_scheduler=arch.action_scheduler,
        num_steps=steps,
        shift=getattr(arch.action_backbone, "shift_action", None) or 5.0,
        shift_video=getattr(arch.video_backbone, "shift_video", None),
        lead=getattr(inf, "lead_modality", "video"),
        alpha=getattr(inf, "variance_shift_alpha", 1.0),
        offset=getattr(inf, "linear_offset", 0.0),
    )


def _gauss_entropy(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    dim = values.shape[1]
    centered = values - values.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(values.shape[0] - 1, 1)
    cov.flat[:: dim + 1] += 1e-8
    _, logdet = np.linalg.slogdet(cov)
    return float(0.5 * (dim * (np.log(2.0 * np.pi) + 1.0) + logdet))


def _frame_stats(chunks: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = {int(start): i for i, start in enumerate(starts)}
    entropy = np.zeros(len(starts), dtype=np.float64)
    spread_mm = np.zeros(len(starts), dtype=np.float64)
    chunk = chunks.shape[2]
    for out_i, frame in enumerate(starts):
        parts = []
        for offset in range(chunk):
            start = int(frame) - offset
            row = index.get(start)
            if row is not None:
                parts.append(chunks[row, :, offset, :])
        samples = np.concatenate(parts, axis=0)
        entropy[out_i] = _gauss_entropy(samples)
        spread_mm[out_i] = float(samples.std(axis=0).mean() * 1000.0)
    return entropy, spread_mm


def _episodes(dataset: Path, selected: str) -> list[dict]:
    wanted = {int(item) for item in selected.split(",") if item}
    rows = []
    with (dataset / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["episode_index"])
            if wanted and index not in wanted:
                continue
            rows.append({
                "episode_index": index,
                "episode_id": row["episode_id"],
                "length": int(row["length"]),
                "task": (row.get("tasks") or [""])[0],
            })
    if wanted and len(rows) != len(wanted):
        found = {row["episode_index"] for row in rows}
        raise ValueError(f"missing episodes: {sorted(wanted - found)}")
    return rows


def _assign(rows: list[dict], stride: int, world: int) -> list[list[dict]]:
    ranked = sorted(rows, key=lambda row: (-(row["length"] + stride - 1) // stride, row["episode_index"]))
    buckets: list[list[dict]] = [[] for _ in range(world)]
    loads = [0 for _ in range(world)]
    for row in ranked:
        rank = min(range(world), key=lambda item: (loads[item], item))
        buckets[rank].append(row)
        loads[rank] += (row["length"] + stride - 1) // stride
    return buckets


def _parquet(dataset: Path, episode_index: int) -> Path:
    return dataset / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _load_tables(dataset: Path, episode_index: int, layout: list[str]):
    columns = ["observation.state_eef", "action_eef", *[f"observation.images.{name}" for name in layout]]
    table = pq.read_table(_parquet(dataset, episode_index), columns=columns)
    images = {name: table[f"observation.images.{name}"].to_pylist() for name in layout}
    return (
        np.asarray(table["observation.state_eef"].to_pylist(), dtype=np.float32),
        np.asarray(table["action_eef"].to_pylist(), dtype=np.float32),
        images,
    )


def _frame_image(dataset: Path, images: dict, layout: list[str], frame: int, height: int, width: int) -> Image.Image:
    cams = {}
    for name in layout:
        with Image.open(dataset / images[name][frame]) as image:
            cams[name] = image.convert("RGB")
    return assemble_multiview_layout(cams, layout, height, width)


def _run_episode(engine, schedule, dataset, row, images_layout, height, width, args, rank: int, progress: dict):
    state, action, images = _load_tables(dataset, row["episode_index"], images_layout)
    starts = np.arange(0, len(state), args.frame_stride, dtype=np.int64)
    action_num = int(getattr(engine.cfg.inference, "num_frames", 33)) - 1
    arch = engine.architecture
    chunks = np.empty((len(starts), args.noises, action_num, 6), dtype=np.float32)
    cursor = 0
    while cursor < len(starts):
        batch_frames = starts[cursor : cursor + args.group]
        conds = [
            _preprocess_case(
                engine,
                _frame_image(dataset, images, images_layout, int(frame), height, width),
                state[int(frame)],
                row["task"],
                args.noises,
                height,
                width,
            )
            for frame in batch_frames
        ]
        inputs = _stack_cases(conds) if len(conds) > 1 else conds[0]
        noises = []
        for frame in batch_frames:
            generator = torch.Generator(device=arch.device)
            generator.manual_seed(row["episode_index"] * 1_000_003 + int(frame))
            noises.append(torch.randn(
                args.noises,
                action_num,
                arch.action_dim,
                device=arch.device,
                dtype=torch.float32,
                generator=generator,
            ))
        latent = _denoise(arch, inputs, torch.cat(noises, dim=0).to(dtype=arch.dtype), schedule)
        actions = np.asarray(arch.normalizer.unnormalize(latent.float().cpu().numpy()), dtype=np.float32)
        actions = actions.reshape(len(batch_frames), args.noises, action_num, -1)
        if actions.shape[-1] != 16:
            raise RuntimeError(f"expected 16-D EEF actions, got {actions.shape}")
        chunks[cursor : cursor + len(batch_frames)] = actions[..., XYZ]
        cursor += len(batch_frames)
        if cursor == len(starts) or cursor % 30 < args.group:
            done_frames = progress["computed"] + cursor
            elapsed = time.perf_counter() - progress["t0"]
            remain = progress["total"] - progress["skipped"] - done_frames
            eta_h = (elapsed / max(done_frames, 1)) * remain / 3600.0
            print(
                f"[entropy] rank={rank} episode {progress['index']}/{progress['episodes']} "
                f"{row['episode_id']} {cursor}/{len(starts)} "
                f"frames {progress['skipped'] + done_frames}/{progress['total']} "
                f"elapsed_h={elapsed / 3600.0:.2f} eta_h={eta_h:.2f}",
                flush=True,
            )
    entropy, spread_mm = _frame_stats(chunks, starts)
    if not np.isfinite(entropy).all():
        raise RuntimeError(f"non-finite entropy in episode {row['episode_index']}")
    return {
        "frame_index": starts,
        "entropy": entropy.astype(np.float32),
        "spread_mm": spread_mm.astype(np.float32),
        "grip": action[starts][:, GRIP].astype(np.float32),
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.noises < 2 or args.group < 1 or args.frame_stride < 1:
        raise ValueError("--noises must be >= 2 and --group/--frame-stride must be >= 1")
    rank, local_rank, world = _rank_world()
    dataset = Path(args.dataset_dir)
    rows = _episodes(dataset, args.episodes)
    buckets = _assign(rows, args.frame_stride, world)
    if args.dry_run:
        for item, bucket in enumerate(buckets):
            sampled = sum((row["length"] + args.frame_stride - 1) // args.frame_stride for row in bucket)
            print(f"[entropy] rank={item} episodes={len(bucket)} sampled_frames={sampled}", flush=True)
        return
    output = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / "action_entropy_fittings_eef_step18185"
    )
    output.mkdir(parents=True, exist_ok=True)
    if local_rank:
        time.sleep(local_rank * 15)
    device = _device(local_rank)
    print(
        f"[entropy] rank={rank}/{world} local_rank={local_rank} loading {args.checkpoint_name} on {device}",
        flush=True,
    )
    cfg, architecture = load_from_checkpoint_dir(args.checkpoint_dir, device=device, ckpt_name=args.checkpoint_name)
    saved = OmegaConf.load(Path(args.checkpoint_dir) / "config.yaml")
    height = int(saved.dataloader.height)
    width = int(saved.dataloader.width)
    layout = [str(name) for name in saved.dataloader.camera_layout]
    deploy_cfg = _load_deploy_yaml()
    deploy_cfg.inference.denoise_steps = args.denoise_steps
    deploy_cfg.optimization.compile.enabled = False
    deploy_cfg.optimization.decode_video = False
    deploy_cfg.optimization.dit_cache.enabled = False
    engine = JointInferenceEngine(cfg=merge_deploy_cfg(cfg, deploy_cfg), architecture=architecture)
    schedule = _make_schedule(engine, args.denoise_steps)
    rank_rows = buckets[rank]
    progress = {
        "computed": 0,
        "skipped": 0,
        "total": sum((row["length"] + args.frame_stride - 1) // args.frame_stride for row in rank_rows),
        "t0": time.perf_counter(),
        "index": 1,
        "episodes": len(rank_rows),
    }
    print(
        f"[entropy] rank={rank}/{world} episodes={len(rank_rows)} sampled_frames={progress['total']}",
        flush=True,
    )
    done = []
    for row in rank_rows:
        sampled = (row["length"] + args.frame_stride - 1) // args.frame_stride
        destination = output / f"episode_{row['episode_index']:06d}.npz"
        if destination.is_file():
            with np.load(destination) as previous:
                same = int(previous["frame_stride"]) == args.frame_stride and int(previous["noises"]) == args.noises
            if same:
                print(f"[entropy] rank={rank} skip existing {destination.name}", flush=True)
                progress["skipped"] += sampled
                progress["index"] += 1
                done.append(row["episode_index"])
                continue
            print(f"[entropy] rank={rank} replace {destination.name} with stride={args.frame_stride}", flush=True)
        print(
            f"[entropy] rank={rank} start {progress['index']}/{progress['episodes']} "
            f"{row['episode_id']} frames={row['length']}",
            flush=True,
        )
        with torch.inference_mode():
            result = _run_episode(engine, schedule, dataset, row, layout, height, width, args, rank, progress)
        progress["computed"] += sampled
        tmp = output / f".episode_{row['episode_index']:06d}.partial.npz"
        np.savez_compressed(
            tmp,
            frame_index=result["frame_index"],
            entropy=result["entropy"],
            spread_mm=result["spread_mm"],
            grip=result["grip"],
            episode_index=row["episode_index"],
            episode_id=row["episode_id"],
            task=row["task"],
            frame_stride=args.frame_stride,
            noises=args.noises,
        )
        tmp.replace(destination)
        done.append(row["episode_index"])
        print(
            f"[entropy] rank={rank} saved {destination.name} "
            f"entropy[{float(result['entropy'].min()):.2f},{float(result['entropy'].max()):.2f}]",
            flush=True,
        )
    manifest = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "episodes": done,
        "frame_stride": args.frame_stride,
        "noises": args.noises,
        "group": args.group,
        "checkpoint": str(Path(args.checkpoint_dir) / args.checkpoint_name),
        "dataset": str(dataset),
    }
    (output / f"rank_{rank:02d}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[entropy] rank={rank} done episodes={len(done)}", flush=True)


if __name__ == "__main__":
    main()
