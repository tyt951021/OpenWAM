"""Evaluate one OpenWAM checkpoint on every Linglong test episode.

The model is loaded once, while each test episode is evaluated with
non-overlapping action chunks (normally stride 32).  Proprio and action
targets follow the checkpoint dataloader: joint columns, or the 16-D EEF
columns when ``state_key`` is ``observation.state_eef``.  Per-episode metrics
and optional GT/generated MP4s are written below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _closed_loop_proprio(action: np.ndarray, state_key: str) -> np.ndarray:
    """Hold the last predicted action as the next chunk's measured state.

    The 32-action chunk ends one 15 Hz step before the next video frame, which
    is the following chunk's first image. EEF wxyz blocks are re-normalized.
    """
    proprio = np.array(action[-1], dtype=np.float32, copy=True)
    if state_key != "observation.state_eef":
        return proprio
    if proprio.shape != (16,):
        raise ValueError(f"EEF closed-loop proprio must be 16-D, got {proprio.shape}")
    for quat_start in (3, 11):
        quat = proprio[quat_start:quat_start + 4]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-6:
            raise ValueError("closed-loop EEF quaternion is degenerate")
        proprio[quat_start:quat_start + 4] = quat / norm
    return proprio


def _install_demospeedup_windows(ds, cfg, entropy_dir: str, offset: int, chunk_stride: int) -> dict:
    """Replace contiguous test windows with one training phase of kept frames.

    The threshold is fit on every episode in this split, matching training.
    Open-loop chunks then advance by ``chunk_stride`` along that kept sequence,
    the same non-overlapping step the contiguous evaluation uses on raw frames.
    """
    from openwam.dataloader.demospeedup import build_keep_plan

    dataloader = cfg.dataloader
    num_offsets = OmegaConf.select(dataloader, "demospeedup_num_offsets", default=None)
    plan = build_keep_plan(
        ds._episodes,
        ds._tables,
        Path(entropy_dir),
        r_precision=int(dataloader.demospeedup_r_precision),
        r_casual=int(dataloader.demospeedup_r_casual),
        num_offsets=None if num_offsets is None else int(num_offsets),
        gripper_radius=int(dataloader.demospeedup_gripper_radius),
        gripper_delta=float(dataloader.demospeedup_gripper_delta),
        smooth_width=int(dataloader.demospeedup_smooth_width),
        min_frames=ds.num_frames,
    )
    if offset < 0 or offset >= int(plan.summary["num_offsets"]):
        raise ValueError(
            f"demospeedup offset {offset} is outside [0, {int(plan.summary['num_offsets']) - 1}]"
        )
    windows = []
    for local, phases in enumerate(plan.tables):
        kept = phases[offset]
        max_start = len(kept) - ds.num_frames
        if max_start >= 0:
            windows.extend((local, start, offset) for start in range(0, max_start + 1, chunk_stride))
    if not windows:
        raise ValueError(f"DemoSpeedup offset {offset} produced no {ds.num_frames}-frame windows")
    ds._kept = plan.tables
    ds._windows = windows
    summary = dict(plan.summary)
    summary["eval_offset"] = int(offset)
    summary["eval_windows"] = len(windows)
    summary["entropy_dir"] = str(entropy_dir)
    print(
        "[eval] demospeedup "
        f"offset={offset} r_precision={summary['r_precision']} r_casual={summary['r_casual']} "
        f"threshold={summary['threshold']:.6f} windows={len(windows)} "
        f"precision_frames={summary['precision_frames']}/{summary['source_frames']}",
        flush=True,
    )
    return summary


def _episode_window_indices(ds, episode: int, chunk_stride: int, limit: int) -> list[int]:
    chosen = []
    for index, window in enumerate(ds._windows):
        if int(window[0]) != episode or int(window[1]) % chunk_stride != 0:
            continue
        chosen.append(index)
        if len(chosen) >= limit:
            break
    return chosen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episode", type=int, default=None, help="Evaluate one test episode index. Default: every episode.")
    p.add_argument("--chunk-stride", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=1000)
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decode-video", action="store_true")
    p.add_argument("--prompt", default=None,
                   help="Replace the episode prompt. Use this when the test bucket wording differs from training.")
    p.add_argument("--closed-loop", action="store_true",
                   help="Feed the previous chunk's last generated frame and last predicted action into the next chunk. Implies --decode-video and requires chunk stride 32.")
    p.add_argument("--demospeedup-entropy-dir", default=None,
                   help="Test-split action entropy. Required when the checkpoint was trained with DemoSpeedup. Rates come from the checkpoint.")
    p.add_argument("--demospeedup-offset", type=int, default=0,
                   help="Which training phase to roll out. 0 keeps residue 0 inside each precision/casual run.")
    args = p.parse_args()
    if args.closed_loop:
        args.decode_video = True
    if args.chunk_stride <= 0 or args.max_windows <= 0:
        p.error("chunk-stride and max-windows must be positive")

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from openwam.dataloader.linglong import LinglongDataset
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import _load_deploy_yaml, merge_deploy_cfg

    print(f"[eval] loading checkpoint {args.checkpoint_dir}", flush=True)
    cfg, architecture = load_from_checkpoint_dir(args.checkpoint_dir, device="cuda:0")
    deploy_cfg = _load_deploy_yaml()
    deploy_cfg.inference.denoise_steps = int(args.denoise_steps)
    deploy_cfg.optimization.compile.enabled = False
    deploy_cfg.optimization.decode_video = bool(args.decode_video)
    cfg = merge_deploy_cfg(cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)

    dl_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
    dl_cfg.dataset_dir = args.dataset_dir
    dl_cfg.overfit_single = False
    dl_cfg.overfit_repeats = 1
    dl_cfg.window_stride = args.chunk_stride
    ds = LinglongDataset.from_config(dl_cfg, split="test")
    if ds.dataset_dir.resolve() != Path(args.dataset_dir).resolve():
        raise ValueError(f"Unexpected dataset override: {ds.dataset_dir}")
    trained_demospeedup = bool(OmegaConf.select(cfg, "dataloader.demospeedup_enabled", default=False))
    if trained_demospeedup and not args.demospeedup_entropy_dir:
        raise ValueError(
            "This checkpoint was trained with DemoSpeedup. Pass --demospeedup-entropy-dir "
            "for the evaluated split so windows use that model's precision/casual rates."
        )
    if args.demospeedup_entropy_dir and not trained_demospeedup:
        raise ValueError("This checkpoint was trained on contiguous frames; do not pass --demospeedup-entropy-dir")
    if args.demospeedup_entropy_dir and args.closed_loop:
        raise ValueError("DemoSpeedup frame gathering is implemented for open-loop evaluation")
    demospeedup_summary = None
    if args.demospeedup_entropy_dir:
        demospeedup_summary = _install_demospeedup_windows(
            ds, cfg, args.demospeedup_entropy_dir, int(args.demospeedup_offset), int(args.chunk_stride)
        )
    if args.closed_loop and args.chunk_stride != ds.num_frames - 1:
        raise ValueError(
            f"closed-loop requires --chunk-stride {ds.num_frames - 1} "
            "so the last generated frame is the next chunk start"
        )
    manifest = json.loads((ds.dataset_dir / "manifest.json").read_text())
    raw_fps = float(manifest["fps"])
    video_fps = raw_fps / ds.video_stride
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_err = []
    all_pred = []
    all_target = []
    episode_results = []
    valid = ds.raw_action_mask
    print(
        f"[eval] state_key={ds.state_key} action_key={ds.action_key} valid_dims={int(valid.sum())} "
        f"closed_loop={bool(args.closed_loop)} demospeedup={bool(args.demospeedup_entropy_dir)}",
        flush=True,
    )

    if args.episode is not None and not 0 <= args.episode < len(ds._episodes):
        raise ValueError(f"episode {args.episode} is outside [0, {len(ds._episodes) - 1}]")
    episode_ids = [args.episode] if args.episode is not None else range(len(ds._episodes))
    for episode in episode_ids:
        window_indices = _episode_window_indices(ds, episode, int(args.chunk_stride), int(args.max_windows))
        if not window_indices:
            print(f"[eval] episode={episode} has no valid windows; skipping", flush=True)
            continue
        window_frames = [ds._window_frames(index) for index in window_indices]
        starts = [int(frames[0]) for _, _, frames in window_frames]
        bucket_prompt = (ds._episodes[episode].get("tasks") or [ds._prompt])[0]
        prompt = args.prompt if args.prompt is not None else bucket_prompt
        print(f"[eval] episode={episode:02d} windows={len(starts)} prompt={prompt!r} bucket_prompt={bucket_prompt!r}", flush=True)
        ep_out = out_root / f"episode_{episode:02d}"
        ep_out.mkdir(parents=True, exist_ok=True)
        writers = None
        if args.decode_video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw

            video_dir = ep_out / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            writers = ExitStack()
            atexit.register(writers.close)
            opts = dict(
                fps=video_fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            comparison = writers.enter_context(
                imageio.get_writer(str(video_dir / "gt_vs_generated_all_chunks.mp4"), **opts)
            )
            generated = writers.enter_context(imageio.get_writer(str(video_dir / "generated_all_chunks.mp4"), **opts))

        if args.closed_loop:
            expect = list(range(starts[0], starts[-1] + 1, args.chunk_stride))
            if starts != expect:
                raise ValueError(f"episode {episode} closed-loop windows are not contiguous: {starts[:8]}")
        predictions, targets, latencies = [], [], []
        feedback = None
        generated_label = "Generated (closed loop)" if args.closed_loop else "Generated (recorded first frame)"
        import pyarrow as pa

        for i, (index, start, source_frames) in enumerate(zip(window_indices, starts, (frames for _, _, frames in window_frames)), 1):
            sample = ds[index]
            table = ds._read_table(episode).take(pa.array(np.asarray(source_frames, dtype=np.int64)))
            raw_state = np.asarray(table[ds.state_key][0].as_py(), dtype=np.float32)
            raw_target = np.asarray(table[ds.action_key].to_pylist(), dtype=np.float32)[: ds.num_frames - 1]
            first_image = [feedback["image"]] if feedback is not None else sample["first_frame_image"]
            proprio = feedback["proprio"] if feedback is not None else raw_state
            t0 = time.perf_counter()
            result = engine.generate(
                {
                    "prompt": prompt,
                    "first_frame_image": first_image,
                    "proprio": proprio,
                    "seed": int(args.seed),
                    "denoise_steps": int(args.denoise_steps),
                }
            )
            latency = (time.perf_counter() - t0) * 1000.0
            pred = np.asarray(result["actions"], dtype=np.float32)
            if pred.shape != raw_target.shape:
                raise RuntimeError(f"shape mismatch episode={episode} start={start}: {pred.shape} vs {raw_target.shape}")
            if args.closed_loop:
                closed_video = result.get("video")
                if not closed_video or len(closed_video) != len(sample["video"]):
                    raise RuntimeError("closed-loop generated video missing or frame count differs from GT")
                feedback = {
                    "image": closed_video[-1].convert("RGB"),
                    "proprio": _closed_loop_proprio(pred, ds.state_key),
                }
            predictions.append(pred)
            targets.append(raw_target)
            latencies.append(latency)
            if args.decode_video:
                video = result.get("video")
                if video is None or len(video) != len(sample["video"]):
                    raise RuntimeError("Generated video missing or frame count differs from GT")
                for j, (gt, frame) in enumerate(zip(sample["video"], video)):
                    if args.chunk_stride == ds.num_frames - 1 and i > 1 and j == 0:
                        continue
                    frame = frame.convert("RGB")
                    if frame.size != gt.size:
                        raise RuntimeError(f"Generated/GT size mismatch: {frame.size}, {gt.size}")
                    w, h = frame.size
                    panel = Image.new("RGB", (w * 2, h + 56), "black")
                    panel.paste(gt, (0, 56))
                    panel.paste(frame, (w, 56))
                    draw = ImageDraw.Draw(panel)
                    raw_frame = int(source_frames[j * ds.video_stride])
                    draw.text((8, 5), "GT (recorded)", fill="white")
                    draw.text((w + 8, 5), generated_label, fill="white")
                    draw.text((8, 23), f"Episode {episode} | chunk {i}/{len(starts)} | start={start} | frame={raw_frame}", fill="white")
                    if demospeedup_summary is None:
                        mode = "Closed-loop image and action" if args.closed_loop else "Non-overlap chunks"
                    else:
                        mode = (
                            f"DemoSpeedup r={demospeedup_summary['r_precision']}/"
                            f"{demospeedup_summary['r_casual']} offset={demospeedup_summary['eval_offset']}"
                        )
                    draw.text((8, 39), f"{mode} | video {video_fps:g} fps | denoise={args.denoise_steps}", fill="white")
                    comparison.append_data(np.asarray(panel))
                    generated.append_data(np.asarray(frame))
            err = pred[:, valid] - raw_target[:, valid]
            print(
                f"[eval] episode={episode:02d} {i:02d}/{len(starts)} start={start:03d} "
                f"latency={latency/1000:.2f}s mae={np.abs(err).mean():.5f}",
                flush=True,
            )

        if writers is not None:
            writers.close()
            atexit.unregister(writers.close)
            (ep_out / "video" / "manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(args.checkpoint_dir),
                        "dataset_dir": str(args.dataset_dir),
                        "episode": episode,
                        "prompt": prompt,
                        "bucket_prompt": bucket_prompt,
                        "chunk_stride": args.chunk_stride,
                        "video_fps": video_fps,
                        "raw_fps": raw_fps,
                        "seed": args.seed,
                        "denoise_steps": args.denoise_steps,
                        "dit_cache_enabled": bool(deploy_cfg.optimization.dit_cache.enabled),
                        "closed_loop": bool(args.closed_loop),
                        "demospeedup": demospeedup_summary,
                        "description": (
                            "The first chunk uses the recorded image and state. Each later chunk uses the previous chunk's last generated frame and last predicted action. That action is one 15 Hz step before the video frame. GT frames stay the recording. Shared boundary frames are removed."
                            if args.closed_loop else
                            "Each chunk uses its recorded first image and state, not autoregressive rollout. Shared boundary frames are removed. DemoSpeedup chunks gather kept frames with the checkpoint's precision and casual rates."
                            if demospeedup_summary is not None else
                            "Each chunk uses its recorded first image and state, not autoregressive rollout. Shared boundary frames removed for contiguous chunks."
                        ),
                        "last_covered_raw_frame": int(window_frames[-1][2][-1]),
                        "omitted_tail_frames": int(ds._episodes[episode]["length"]) - int(window_frames[-1][2][-1]) - 1,
                        "chunks": starts,
                        "source_frames": [frames.tolist() for _, _, frames in window_frames],
                    },
                    indent=2,
                )
            )
        pred_arr = np.stack(predictions)
        target_arr = np.stack(targets)
        err = pred_arr[:, :, valid] - target_arr[:, :, valid]
        metrics = {
            "checkpoint_dir": str(args.checkpoint_dir),
            "dataset_dir": str(args.dataset_dir),
            "episode": episode,
            "prompt": prompt,
            "bucket_prompt": bucket_prompt,
            "windows": len(starts),
            "denoise_steps": args.denoise_steps,
            "state_key": ds.state_key,
            "action_key": ds.action_key,
            "closed_loop": bool(args.closed_loop),
            "demospeedup_r_precision": None if demospeedup_summary is None else int(demospeedup_summary["r_precision"]),
            "demospeedup_r_casual": None if demospeedup_summary is None else int(demospeedup_summary["r_casual"]),
            "demospeedup_offset": None if demospeedup_summary is None else int(demospeedup_summary["eval_offset"]),
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "max_abs_error": float(np.abs(err).max()),
            "per_dim_mae": np.abs(err).mean(axis=(0, 1)).tolist(),
            "latency_ms_mean": float(np.mean(latencies)),
        }
        (ep_out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        np.savez_compressed(ep_out / "metrics.npz", predictions=pred_arr, targets=target_arr, starts=np.asarray(starts))
        episode_results.append(metrics)
        all_err.append(err)
        all_pred.append(pred_arr)
        all_target.append(target_arr)

    joined_err = np.concatenate(all_err, axis=0)
    aggregate = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_dir": str(args.dataset_dir),
        "episodes": len(episode_results),
        "windows": int(sum(x["windows"] for x in episode_results)),
        "denoise_steps": args.denoise_steps,
        "chunk_stride": args.chunk_stride,
        "seed": args.seed,
        "state_key": ds.state_key,
        "action_key": ds.action_key,
        "closed_loop": bool(args.closed_loop),
        "valid_action_dims": int(valid.sum()),
        "dit_cache_enabled": bool(deploy_cfg.optimization.dit_cache.enabled),
        "mae": float(np.abs(joined_err).mean()),
        "rmse": float(np.sqrt(np.mean(joined_err**2))),
        "max_abs_error": float(np.abs(joined_err).max()),
        "per_dim_mae": np.abs(joined_err).mean(axis=(0, 1)).tolist(),
        "episode_metrics": episode_results,
    }
    (out_root / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
