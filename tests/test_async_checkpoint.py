"""Async checkpoint copy: the completion marker lands only after the files do."""

import os
import threading
from pathlib import Path

import pytest

from openwam.train.utils.async_checkpoint import CheckpointPublisher
from openwam.train.utils.checkpointing import find_latest_accel_state


class _Accel:
    is_main_process = True
    is_local_main_process = True
    process_index = 0
    num_processes = 1

    def __init__(self):
        self.saved: list[str] = []

    def wait_for_everyone(self):
        pass

    def save_state(self, state_dir):
        self.saved.append(state_dir)
        os.makedirs(os.path.join(state_dir, "pytorch_model"), exist_ok=True)
        with open(os.path.join(state_dir, "pytorch_model", "shard.bin"), "w") as handle:
            handle.write("optim-state")

    def get_state_dict(self, _model):
        return {"weight": "cpu-snapshot"}

    def unwrap_model(self, model):
        return model


class _Arch:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def save_checkpoint(self, path, state_dict=None):
        if self.fail:
            raise RuntimeError("disk full")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            handle.write(repr(state_dict))


def _publisher(tmp_path, *, enabled=True, min_free_bytes=1, gate=None, staging=True):
    return CheckpointPublisher(
        _Accel(),
        enabled=enabled,
        staging_root=str(tmp_path / "stage") if staging else None,
        min_free_bytes=min_free_bytes,
        copy_gate=gate,
    )


def test_async_copy_publishes_marker_only_after_files(tmp_path):
    gate = threading.Event()
    publisher = _publisher(tmp_path, gate=gate)
    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoint_step_1.safetensors").write_text("old")
    (output / "accel_state_step_1").mkdir()
    (output / "accel_state_step_1" / "trainer_state.json").write_text("{}")
    (output / "accel_state_step_9").mkdir()  # crashed write, no marker

    publisher.publish(
        _Arch(), str(output), 2, 1, 0,
        resume_context={"seed": 1}, final=False, include_full_state=True, prune="rolling", keep_last_k=1,
    )
    state_dir = output / "accel_state_step_2"
    assert not (state_dir / "trainer_state.json").exists()

    gate.set()
    publisher.wait()

    assert (state_dir / "pytorch_model" / "shard.bin").read_text() == "optim-state"
    assert (state_dir / "trainer_state.json").is_file()
    assert find_latest_accel_state(str(output)).endswith("accel_state_step_2")
    assert (output / "checkpoint_step_2.safetensors").is_file()
    assert not (output / "checkpoint_step_1.safetensors").exists()
    assert not (output / "accel_state_step_1").exists()
    assert not (output / "accel_state_step_9").exists()
    assert not any(path.is_file() for path in (tmp_path / "stage").rglob("*"))
    publisher.close()


def test_async_copy_failure_is_raised_on_wait(tmp_path):
    publisher = _publisher(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    publisher.publish(
        _Arch(fail=True), str(output), 4, 1, 0,
        resume_context=None, final=False, include_full_state=True, prune="rolling", keep_last_k=2,
    )
    with pytest.raises(RuntimeError, match="disk full"):
        publisher.wait()
    assert not (output / "accel_state_step_4" / "trainer_state.json").exists()
    publisher.close()


def test_async_falls_back_to_sync_when_staging_is_too_small(tmp_path):
    publisher = _publisher(tmp_path, min_free_bytes=10**18)
    output = tmp_path / "run"
    output.mkdir()
    arch = _Arch()
    publisher.publish(
        arch, str(output), 5, 1, 0,
        resume_context=None, final=True, include_full_state=True, prune="rolling", keep_last_k=2,
    )
    saved = publisher.accelerator.saved
    assert saved and saved[0].startswith(str(output))
    assert (output / "accel_state_step_5" / "trainer_state.json").is_file()
    assert (output / "checkpoint_step_5.safetensors").is_file()
    assert not (tmp_path / "stage").exists() or not any((tmp_path / "stage").rglob("*"))
    publisher.close()


def test_disabled_async_writes_before_publish_returns(tmp_path: Path):
    publisher = _publisher(tmp_path, enabled=False, staging=False)
    output = tmp_path / "run"
    output.mkdir()
    publisher.publish(
        _Arch(), str(output), 6, 2, 1,
        resume_context=None, final=False, include_full_state=True, prune="rolling", keep_last_k=2,
    )
    assert (output / "accel_state_step_6" / "trainer_state.json").is_file()
    assert (output / "checkpoint_step_6.safetensors").is_file()
    publisher.close()
