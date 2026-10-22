"""Offline Piper X inference and GT/generated video export.

Open loop gives every chunk its recorded first image and measured EEF state.
Closed loop gives the first chunk that same observation, then feeds each later
chunk the previous chunk's last generated frame and last predicted action.
The comparison video keeps the recorded frames on the left.

The action target is the 16D absolute EEF in the bucket. Metrics use the
checkpoint's active dimensions. A closed-loop rollout uses chunk stride 32,
which is one 15 Hz step before the next chunk's first video frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


TRAIN_BUCKET = Path("/vePFS/data/piper-x_openwam/piper_x_openwam_train_15hz")
TEST_BUCKET = Path("/vePFS/data/piper-x_openwam/piper_x_openwam_test_15hz")


def closed_loop_proprio(action: np.ndarray) -> np.ndarray:
    """Use the last predicted 16D action as the next chunk's measured state."""
    proprio = np.array(action[-1], dtype=np.float32, copy=True)
    if proprio.shape != (16,):
        raise ValueError(f"Piper X closed-loop proprio must be 16-D, got {proprio.shape}")
    for quat_start in (3, 11):
        quat = proprio[quat_start:quat_start + 4]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-6:
            raise ValueError("closed-loop EEF quaternion is degenerate")
        proprio[quat_start:quat_start + 4] = quat / norm
    return proprio


def episode_starts(windows, episode: int, chunk_stride: int, max_windows: int) -> list[int]:
    starts = [int(start) for ep, start in windows if int(ep) == int(episode) and int(start) % chunk_stride == 0]
    return starts[:max_windows]


def require_contiguous(starts: list[int], chunk_stride: int, episode: int) -> None:
    if not starts:
        raise ValueError(f"episode {episode} has no closed-loop windows")
    expected = list(range(starts[0], starts[-1] + 1, chunk_stride))
    if starts != expected:
        raise ValueError(f"episode {episode} closed-loop windows are not contiguous: {starts[:8]}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Bucket for --split. Defaults to the converted Piper X train or test bucket.")
    parser.add_argument("--episode", type=int, default=None,
                        help="One episode index. Default: episode 0 for train, every episode for test.")
    parser.add_argument("--chunk-stride", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-video", action="store_true", help="Write GT and generated MP4s.")
    parser.add_argument("--closed-loop", action="store_true",
                        help="Roll the generated frame and action into the next chunk. Implies --decode-video and chunk stride 32.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _append_comparison(comparison, generated_writer, sample, video, *, episode, chunk, n_chunks, start,
                       video_stride, raw_fps, video_fps, denoise_steps, seed, closed_loop, drop_boundary):
    from PIL import Image, ImageDraw

    label = "Generated (closed loop)" if closed_loop else "Generated (recorded first frame)"
    mode = "Closed-loop image and action" if closed_loop else "Open-loop recorded first frame"
    for index, (gt, frame) in enumerate(zip(sample["video"], video)):
        if drop_boundary and index == 0:
            continue
        frame = frame.convert("RGB")
        if frame.size != gt.size:
            raise RuntimeError(f"Generated/GT size mismatch: {frame.size}, {gt.size}")
        width, height = frame.size
        panel = Image.new("RGB", (width * 2, height + 56), "black")
        panel.paste(gt, (0, 56))
        panel.paste(frame, (width, 56))
        draw = ImageDraw.Draw(panel)
        raw_frame = start + index * video_stride
        draw.text((8, 5), "GT (recorded)", fill="white")
        draw.text((width + 8, 5), label, fill="white")
        draw.text((8, 23), f"Episode {episode} | chunk {chunk}/{n_chunks} | start={start} | frame={raw_frame} | t={raw_frame / raw_fps:.3f}s", fill="white")
        draw.text((8, 39), f"{mode} | video {video_fps:g} fps | denoise={denoise_steps} | seed={seed}", fill="white")
        comparison.append_data(np.asarray(panel))
        generated_writer.append_data(np.asarray(frame))


def _evaluate_episode(engine, dataset, episode, starts, args, raw_fps, video_fps, deploy_cfg, output_dir: Path):
    from contextlib import ExitStack
    import atexit

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = None
    comparison = generated_writer = None
    if args.decode_video:
        import imageio.v2 as imageio

        video_dir = output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        writers = ExitStack()
        atexit.register(writers.close)
        options = dict(fps=video_fps, codec="libx264", macro_block_size=1,
                       ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        comparison = writers.enter_context(imageio.get_writer(str(video_dir / "gt_vs_generated_all_chunks.mp4"), **options))
        generated_writer = writers.enter_context(imageio.get_writer(str(video_dir / "generated_all_chunks.mp4"), **options))
    predictions, targets, latencies = [], [], []
    feedback = None
    contiguous = args.chunk_stride == dataset.num_frames - 1
    valid = dataset.raw_action_mask
    for index, start in enumerate(starts, 1):
        sample = dataset[dataset._windows.index((episode, start))]
        table = dataset._read_table(episode).slice(start, dataset.num_frames)
        raw_state = np.asarray(table[dataset.state_key][0].as_py(), dtype=np.float32)
        raw_target = np.asarray(table[dataset.action_key].to_pylist(), dtype=np.float32)[:dataset.num_frames - 1]
        first_image = [feedback["image"]] if feedback is not None else sample["first_frame_image"]
        proprio = feedback["proprio"] if feedback is not None else raw_state
        started = time.perf_counter()
        result = engine.generate({
            "prompt": sample["prompt"],
            "first_frame_image": first_image,
            "proprio": proprio,
            "seed": int(args.seed),
            "denoise_steps": int(args.denoise_steps),
        })
        latency = (time.perf_counter() - started) * 1000.0
        prediction = np.asarray(result["actions"], dtype=np.float32)
        if prediction.shape != raw_target.shape:
            raise RuntimeError(f"shape mismatch episode={episode} start={start}: {prediction.shape} vs {raw_target.shape}")
        if args.closed_loop:
            video = result.get("video")
            if not video or len(video) != len(sample["video"]):
                raise RuntimeError("closed-loop generated video missing or frame count differs from GT")
            feedback = {"image": video[-1].convert("RGB"), "proprio": closed_loop_proprio(prediction)}
        if args.decode_video:
            video = result.get("video")
            if video is None or len(video) != len(sample["video"]):
                raise RuntimeError("Generated video missing or frame count differs from GT")
            _append_comparison(
                comparison, generated_writer, sample, video,
                episode=episode, chunk=index, n_chunks=len(starts), start=start,
                video_stride=dataset.video_stride, raw_fps=raw_fps, video_fps=video_fps,
                denoise_steps=args.denoise_steps, seed=args.seed, closed_loop=args.closed_loop,
                drop_boundary=contiguous and index > 1,
            )
        predictions.append(prediction)
        targets.append(raw_target)
        latencies.append(latency)
        error = prediction[:, valid] - raw_target[:, valid]
        print(
            f"[eval] episode={episode:02d} {index:02d}/{len(starts)} start={start:03d} "
            f"latency={latency / 1000:.2f}s mae={np.abs(error).mean():.5f}",
            flush=True,
        )
    if writers is not None:
        writers.close()
        atexit.unregister(writers.close)
        description = (
            "The first chunk uses the recorded image and state. Each later chunk uses the previous chunk's last generated frame and last predicted action. GT frames stay the recording. Shared boundary frames are removed."
            if args.closed_loop else
            "Each chunk uses its recorded first image and measured state. Later recorded frames are only for comparison. Shared boundary frames are removed."
        )
        (output_dir / "video" / "manifest.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint_dir),
            "dataset_dir": str(dataset.dataset_dir),
            "episode": episode,
            "chunk_stride": args.chunk_stride,
            "video_fps": video_fps,
            "raw_fps": raw_fps,
            "seed": args.seed,
            "denoise_steps": args.denoise_steps,
            "dit_cache_enabled": bool(deploy_cfg.optimization.dit_cache.enabled),
            "closed_loop": bool(args.closed_loop),
            "description": description,
            "last_covered_raw_frame": starts[-1] + dataset.num_frames - 1,
            "omitted_tail_frames": int(dataset._episodes[episode]["length"]) - (starts[-1] + dataset.num_frames),
            "chunks": starts,
        }, indent=2))
    predicted = np.stack(predictions)
    target = np.stack(targets)
    error = predicted[:, :, valid] - target[:, :, valid]
    metrics = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_dir": str(dataset.dataset_dir),
        "split": args.split,
        "episode": episode,
        "windows": len(starts),
        "denoise_steps": args.denoise_steps,
        "chunk_stride": args.chunk_stride,
        "state_key": dataset.state_key,
        "action_key": dataset.action_key,
        "closed_loop": bool(args.closed_loop),
        "valid_action_dims": int(valid.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(np.abs(error).max()),
        "per_dim_mae": np.abs(error).mean(axis=(0, 1)).tolist(),
        "latency_ms_mean": float(np.mean(latencies)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    np.savez_compressed(output_dir / "metrics.npz", predictions=predicted, targets=target, starts=np.asarray(starts))
    return metrics, error


def main() -> None:
    args = _build_parser().parse_args()
    if args.closed_loop:
        args.decode_video = True
    if args.chunk_stride <= 0 or args.max_windows <= 0:
        raise ValueError("chunk-stride and max-windows must be positive")
    if args.dataset_dir is None:
        args.dataset_dir = TEST_BUCKET if args.split == "test" else TRAIN_BUCKET

    repo = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo))
    from omegaconf import OmegaConf

    from openwam.dataloader.piper_x import PiperXDataset
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import _load_deploy_yaml, merge_deploy_cfg

    print(f"[eval] loading checkpoint {args.checkpoint_dir}", flush=True)
    cfg, architecture = load_from_checkpoint_dir(str(args.checkpoint_dir), device="cuda:0")
    deploy_cfg = _load_deploy_yaml()
    deploy_cfg.inference.denoise_steps = int(args.denoise_steps)
    deploy_cfg.optimization.compile.enabled = False
    deploy_cfg.optimization.decode_video = bool(args.decode_video)
    cfg = merge_deploy_cfg(cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)

    dl_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
    dl_cfg.dataset_dir = str(args.dataset_dir)
    dl_cfg.overfit_single = False
    dl_cfg.overfit_repeats = 1
    dl_cfg.window_stride = int(args.chunk_stride)
    dataset = PiperXDataset.from_config(dl_cfg, split=args.split)
    if args.closed_loop and args.chunk_stride != dataset.num_frames - 1:
        raise ValueError(
            f"closed-loop requires --chunk-stride {dataset.num_frames - 1} "
            "so the last generated frame is the next chunk start"
        )
    manifest = json.loads((dataset.dataset_dir / "manifest.json").read_text())
    raw_fps = float(manifest["fps"])
    video_fps = raw_fps / dataset.video_stride
    if args.episode is None:
        episodes = [0] if args.split == "train" else list(range(len(dataset._episodes)))
    else:
        if not 0 <= args.episode < len(dataset._episodes):
            raise ValueError(f"episode {args.episode} is outside [0, {len(dataset._episodes) - 1}]")
        episodes = [args.episode]

    print(
        f"[eval] split={args.split} dataset={dataset.dataset_dir} episodes={episodes} "
        f"valid_dims={int(dataset.raw_action_mask.sum())} closed_loop={bool(args.closed_loop)}",
        flush=True,
    )
    results = []
    errors = []
    for episode in episodes:
        starts = episode_starts(dataset._windows, episode, args.chunk_stride, args.max_windows)
        if not starts:
            print(f"[eval] episode={episode} has no windows at stride {args.chunk_stride}; skipping", flush=True)
            continue
        if args.closed_loop:
            require_contiguous(starts, args.chunk_stride, episode)
        print(f"[eval] episode={episode:02d} windows={len(starts)}", flush=True)
        metrics, error = _evaluate_episode(
            engine, dataset, episode, starts, args, raw_fps, video_fps, deploy_cfg,
            args.output_dir / f"episode_{episode:02d}",
        )
        results.append(metrics)
        errors.append(error)
    if not results:
        raise ValueError("no episodes produced a window")
    joined = np.concatenate(errors, axis=0)
    aggregate = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_dir": str(dataset.dataset_dir),
        "split": args.split,
        "episodes": len(results),
        "windows": int(sum(item["windows"] for item in results)),
        "denoise_steps": args.denoise_steps,
        "chunk_stride": args.chunk_stride,
        "seed": args.seed,
        "state_key": dataset.state_key,
        "action_key": dataset.action_key,
        "closed_loop": bool(args.closed_loop),
        "decode_video": bool(args.decode_video),
        "valid_action_dims": int(dataset.raw_action_mask.sum()),
        "mae": float(np.abs(joined).mean()),
        "rmse": float(np.sqrt(np.mean(joined ** 2))),
        "max_abs_error": float(np.abs(joined).max()),
        "per_dim_mae": np.abs(joined).mean(axis=(0, 1)).tolist(),
        "episode_metrics": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in aggregate.items() if key != "episode_metrics"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
