"""Background copy of checkpoints off the training step.

DeepSpeed's ``save_state`` is a collective, so it stays on the training
threads. It writes to a node-local staging directory (``/dev/shm`` when that
has room). Each node's local rank 0 then copies its files to the run directory
while training continues. The completion marker is written only after every
node has finished, so resume never reads a partial copy.

The weights file is snapshotted to CPU before training resumes and written by
the same background thread. A copy still in flight when the next save is due
is finished first, which keeps staging from stacking. The final save waits for
the copy before the process exits.
"""

import hashlib
import logging
import os
import shutil
import threading
import time
import traceback

from openwam.train.utils.checkpointing import (
    finalize_keep_weights_only,
    manage_checkpoints,
    save_full_state,
    save_weights,
    write_trainer_state,
    write_weights_atomic,
)

logger = logging.getLogger(__name__)

# One full state is ~103 GiB plus the rank-0 weights clone and the buffer
# torch.save holds while it fills staging. Below this, fall back to writing
# the checkpoint directly so a tight node does not OOM.
_DEFAULT_MIN_FREE_BYTES = 200 * (1024**3)
_MARKER_WAIT_SECONDS = 60 * 60
_MARKER_POLL_SECONDS = 30


def _mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _on_tmpfs(path: str) -> bool:
    real = os.path.realpath(path)
    shm = os.path.realpath("/dev/shm")
    return real == shm or real.startswith(shm + os.sep)


def _announce(msg: str) -> None:
    logger.info(msg)
    print(msg, flush=True)


def _failure_path(output_path: str, step: int) -> str:
    return os.path.join(output_path, ".ckpt_copy_status", f"step_{step}.failed")


def _node_marker(output_path: str, step: int, node_rank: int) -> str:
    return os.path.join(output_path, ".ckpt_copy_status", f"step_{step}", f"node_{node_rank}.done")


def _write_text_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _copy_tree(src: str, dst: str) -> None:
    """Copy files into ``dst``, replacing each one atomically, then drop ``src``."""
    os.makedirs(dst, exist_ok=True)
    for dirpath, _dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target_dir = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target_dir, exist_ok=True)
        for name in filenames:
            if name == "trainer_state.json" or name.endswith(".partial") or name.endswith(".tmp"):
                continue
            src_file = os.path.join(dirpath, name)
            dst_file = os.path.join(target_dir, name)
            partial = dst_file + ".partial"
            shutil.copyfile(src_file, partial)
            os.replace(partial, dst_file)
            os.remove(src_file)


def _wait_for_node_markers(output_path: str, step: int, num_nodes: int) -> None:
    deadline = time.monotonic() + _MARKER_WAIT_SECONDS
    next_log = time.monotonic()
    while True:
        if os.path.isfile(_failure_path(output_path, step)):
            raise RuntimeError(f"checkpoint copy failed on another rank at step {step}")
        ready = sum(1 for node in range(num_nodes) if os.path.isfile(_node_marker(output_path, step, node)))
        if ready >= num_nodes:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {num_nodes} checkpoint copy markers "
                f"({ready} ready) at step {step}"
            )
        now = time.monotonic()
        if now >= next_log:
            _announce(f"[checkpoint] step {step} waiting for node copies ({ready}/{num_nodes})")
            next_log = now + _MARKER_POLL_SECONDS
        time.sleep(1)


class CheckpointPublisher:
    """Publish a checkpoint without holding the training step on the filesystem write."""

    def __init__(
        self,
        accelerator,
        *,
        enabled: bool,
        staging_root: str | None = None,
        min_free_bytes: int | None = None,
        copy_gate: threading.Event | None = None,
    ):
        self.accelerator = accelerator
        self.enabled = bool(enabled)
        self._configured_root = staging_root or None
        self._min_free_bytes = _DEFAULT_MIN_FREE_BYTES if min_free_bytes is None else int(min_free_bytes)
        self._copy_gate = copy_gate
        self._thread: threading.Thread | None = None
        self._inflight_step: int | None = None
        self._inflight_output: str | None = None
        self._announced = False
        process_index = int(getattr(accelerator, "process_index", 0))
        num_processes = int(getattr(accelerator, "num_processes", 1))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or "0")
        if local_world <= 0:
            local_world = num_processes
        self._num_nodes = max(1, num_processes // local_world)
        self._node_rank = process_index // local_world
        self._is_main = bool(accelerator.is_main_process)
        self._is_local_main = bool(getattr(accelerator, "is_local_main_process", self._is_main))

    def publish(
        self,
        architecture,
        output_path: str,
        global_step: int,
        opt_step: int,
        epoch: int,
        *,
        resume_context,
        final: bool,
        include_full_state: bool,
        prune: str,
        keep_last_k: int,
    ) -> None:
        """Stage this step and return. The previous copy is finished first."""
        self.wait()
        use_async, staging_root, reason = self._async_decision(include_full_state)
        if not self._announced and self._is_main:
            self._announced = True
            if self.enabled and use_async:
                where = staging_root if include_full_state else "cpu weights snapshot"
                _announce(f"[checkpoint] async saves on; staging={where} nodes={self._num_nodes}")
            elif self.enabled:
                _announce(f"[checkpoint] async saves off for this run ({reason})")
        if not use_async:
            self._publish_sync(
                architecture, output_path, global_step, opt_step, epoch,
                resume_context=resume_context, final=final, include_full_state=include_full_state,
                prune=prune, keep_last_k=keep_last_k,
            )
            return

        if self._is_main:
            stale_failure = _failure_path(output_path, global_step)
            if os.path.isfile(stale_failure):
                os.remove(stale_failure)
        self.accelerator.wait_for_everyone()

        state_dict = None
        if include_full_state:
            if not staging_root:
                raise RuntimeError("async checkpoint staging root is missing")
            staging_dir = os.path.join(
                staging_root,
                "openwam_" + hashlib.sha1(output_path.encode()).hexdigest()[:8],
                f"step_{global_step}",
            )
            if self._is_local_main and os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir)
            self.accelerator.wait_for_everyone()
            if self._is_main:
                os.makedirs(os.path.dirname(staging_dir), exist_ok=True)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(staging_dir)
            self.accelerator.wait_for_everyone()
        else:
            staging_dir = None

        snapshot_error = None
        if self._is_main:
            try:
                state_dict = self.accelerator.get_state_dict(architecture)
                architecture = self.accelerator.unwrap_model(architecture)
            except Exception:
                snapshot_error = traceback.format_exc()
                _write_text_atomic(_failure_path(output_path, global_step), snapshot_error)
        self.accelerator.wait_for_everyone()
        if snapshot_error is not None or os.path.isfile(_failure_path(output_path, global_step)):
            if staging_dir and self._is_local_main and os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            with open(_failure_path(output_path, global_step)) as handle:
                detail = handle.read().strip()
            raise RuntimeError(f"checkpoint snapshot failed at step {global_step}: {detail}")

        self._inflight_step = int(global_step)
        self._inflight_output = output_path
        if self._is_local_main and (staging_dir is not None or state_dict is not None):
            worker = threading.Thread(
                target=self._worker,
                name=f"ckpt-copy-{global_step}",
                args=(
                    architecture, state_dict, staging_dir, output_path, int(global_step),
                    int(opt_step), int(epoch), resume_context, final, prune, int(keep_last_k),
                ),
                daemon=False,
            )
            self._thread = worker
            worker.start()
            if self._is_main:
                _announce(
                    f"[checkpoint] step {global_step} staged locally; "
                    f"{'final ' if final else ''}copy continues in the background"
                )

    def wait(self) -> None:
        """Block until the copy started on this rank has finished.

        Every rank enters. A failed copy raises on every rank once the writing
        rank has recorded it.
        """
        prev_step = self._inflight_step
        prev_output = self._inflight_output
        self._join_local_thread()
        self.accelerator.wait_for_everyone()
        if prev_step is None or prev_output is None:
            return
        fail_path = _failure_path(prev_output, prev_step)
        if os.path.isfile(fail_path):
            with open(fail_path) as handle:
                detail = handle.read().strip()
            raise RuntimeError(f"async checkpoint copy failed at step {prev_step}: {detail}")
        self._inflight_step = None
        self._inflight_output = None

    def raise_if_async_failed(self) -> None:
        """Stop training soon after a background copy fails.

        Local check only, so it does not add a collective to every step. Ranks
        observe the same failure file and leave the loop; the next collective
        is not required to surface the error.
        """
        if self._inflight_step is None or self._inflight_output is None:
            return
        fail_path = _failure_path(self._inflight_output, self._inflight_step)
        if not os.path.isfile(fail_path):
            return
        with open(fail_path) as handle:
            detail = handle.read().strip()
        raise RuntimeError(f"async checkpoint copy failed at step {self._inflight_step}: {detail}")

    def close(self) -> None:
        """Finish this rank's copy without a collective. Used while the process is exiting."""
        self._join_local_thread()

    def _join_local_thread(self) -> None:
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join()

    def _async_decision(self, include_full_state: bool) -> tuple[bool, str | None, str]:
        if not self.enabled:
            return False, None, "async_checkpoints is false"
        if not include_full_state:
            return True, self._configured_root, ""
        root = self._configured_root or "/dev/shm"
        if not os.path.isdir(root):
            if self._configured_root:
                try:
                    os.makedirs(root, exist_ok=True)
                except OSError as exc:
                    return False, None, f"cannot create staging dir {root}: {exc}"
            else:
                return False, None, "/dev/shm is not available"
        try:
            free = shutil.disk_usage(root).free
        except OSError as exc:
            return False, None, f"cannot stat {root}: {exc}"
        if free < self._min_free_bytes:
            return False, None, (
                f"{root} has {free / 2**30:.0f} GiB free, need {self._min_free_bytes / 2**30:.0f} GiB"
            )
        if _on_tmpfs(root):
            mem = _mem_available_bytes()
            if mem is not None and mem < self._min_free_bytes:
                return False, None, (
                    f"MemAvailable is {mem / 2**30:.0f} GiB, need {self._min_free_bytes / 2**30:.0f} GiB "
                    "to stage the checkpoint in memory"
                )
        return True, root, ""

    def _publish_sync(
        self, architecture, output_path, global_step, opt_step, epoch, *,
        resume_context, final, include_full_state, prune, keep_last_k,
    ) -> None:
        save_weights(self.accelerator, architecture, output_path, global_step, final=final)
        if include_full_state:
            save_full_state(
                self.accelerator, output_path, global_step, opt_step, epoch, resume_context=resume_context,
            )
        else:
            self.accelerator.wait_for_everyone()
        if self._is_main:
            self._prune(output_path, prune, keep_last_k)
        self.accelerator.wait_for_everyone()

    def _worker(
        self, architecture, state_dict, staging_dir, output_path, global_step, opt_step, epoch,
        resume_context, final, prune, keep_last_k,
    ) -> None:
        try:
            if self._copy_gate is not None:
                self._copy_gate.wait()
            started = time.monotonic()
            state_dir = os.path.join(output_path, f"accel_state_step_{global_step}")
            if staging_dir is not None:
                if self._is_main:
                    marker = os.path.join(state_dir, "trainer_state.json")
                    if os.path.exists(marker):
                        os.remove(marker)
                _copy_tree(staging_dir, state_dir)
            if state_dict is not None:
                path = write_weights_atomic(architecture, output_path, global_step, state_dict)
                state_dict = None
                _announce(
                    f"[checkpoint] {'final ' if final else ''}weights copied: {path}"
                )
            if staging_dir is not None:
                _write_text_atomic(_node_marker(output_path, global_step, self._node_rank), "ok\n")
            if self._is_main:
                if staging_dir is not None:
                    _wait_for_node_markers(output_path, global_step, self._num_nodes)
                    write_trainer_state(
                        state_dir, global_step=global_step, opt_step=opt_step, epoch=epoch,
                        resume_context=resume_context,
                    )
                    status_dir = os.path.join(output_path, ".ckpt_copy_status", f"step_{global_step}")
                    shutil.rmtree(status_dir, ignore_errors=True)
                self._prune(output_path, prune, keep_last_k)
                _announce(
                    f"[checkpoint] step {global_step} background copy finished "
                    f"in {time.monotonic() - started:.0f}s"
                )
        except Exception:
            detail = traceback.format_exc()
            try:
                _write_text_atomic(_failure_path(output_path, global_step), detail)
            except OSError:
                logger.exception("Failed to record checkpoint copy error")
            _announce(f"[checkpoint] step {global_step} background copy failed\n{detail}")
        finally:
            if staging_dir and os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)

    @staticmethod
    def _prune(output_path: str, prune: str, keep_last_k: int) -> None:
        if prune == "weights_only":
            finalize_keep_weights_only(output_path, keep_last_k)
        elif prune == "final":
            # Finished run: keep the recent weight files, and only the newest resume state.
            manage_checkpoints(output_path, keep_last_k, keep_last_state=1)
        else:
            manage_checkpoints(output_path, keep_last_k)
