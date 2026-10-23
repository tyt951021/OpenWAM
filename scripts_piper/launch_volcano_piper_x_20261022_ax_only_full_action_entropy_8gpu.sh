#!/usr/bin/env bash
# One worker, eight GPUs. Per-frame EEF action entropy for the 20261022
# A/X-only Piper X train bucket. Probe is the 10-epoch finetune at step 18560.
# Stride 2 and eight noises match the fittings DemoSpeedup entropy. The 16D
# layout is left xyz+wxyz+gripper, then right, so this reuses the existing
# entropy runner.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$REPO_ROOT/scripts/activate_vepfs_openwam.sh"
# The shared activate script points at another checkout. This recipe needs the
# entropy runner in the checkout that contains this launcher.
export OPENWAM_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-$CONDA_PREFIX/bin/python}"
DATASET_DIR=/vePFS/data/piper-x_openwam/20261022_ax_only_full/piper_x_openwam_train_15hz
CKPT_DIR=/vePFS/zhuoyu/OpenWAM/outputs/piper_x_20261022_ax_only_full_eef_8gpu_e10_bs4_official_finetune/2026-10-22_19-49-12
CKPT_NAME=checkpoint_step_18560.safetensors
FRAME_STRIDE="${FRAME_STRIDE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$OPENWAM_REPO_ROOT/outputs/action_entropy_piper_x_20261022_ax_only_full_step18560_stride${FRAME_STRIDE}}"
if [[ ! "$FRAME_STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: FRAME_STRIDE must be a positive integer" >&2
  exit 2
fi
if [[ "${NNODES:-1}" != 1 ]]; then
  echo "ERROR: this recipe requires one worker with eight GPUs" >&2
  exit 2
fi
role="${MLP_ROLE:-worker}"
upper_role="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
hosts_var="MLP_${upper_role}_ALL_HOSTS"
if [[ ! "$hosts_var" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
  echo "ERROR: invalid MLP_ROLE" >&2
  exit 2
fi
hosts="${!hosts_var:-${MLP_WORKER_ALL_HOSTS:-}}"
hosts="${hosts//,/ }"
host_list=( $hosts )
if (( ${#host_list[@]} > 1 )); then
  echo "ERROR: select one worker with eight GPUs for this recipe" >&2
  exit 2
fi
for artifact in \
  "$DATASET_DIR/manifest.json" \
  "$DATASET_DIR/meta/episodes.jsonl" \
  "$DATASET_DIR/meta/state_eef_stats.json" \
  "$DATASET_DIR/meta/action_eef_stats.json" \
  "$CKPT_DIR/config.yaml" \
  "$CKPT_DIR/$CKPT_NAME"
do
  if [[ ! -f "$artifact" ]]; then
    echo "ERROR: missing artifact: $artifact" >&2
    exit 2
  fi
done
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" != 1 ]]; then
  mkdir -p "$OPENWAM_REPO_ROOT/logs/action_entropy_piper_x_20261022_ax_only_full_step18560" "$OUTPUT_DIR"
  log_path="$OPENWAM_REPO_ROOT/logs/action_entropy_piper_x_20261022_ax_only_full_step18560/launch_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
  exec > >(tee -a "$log_path") 2>&1
  echo "OpenWAM Piper X action-entropy log: $log_path"
fi
echo "dataset=$DATASET_DIR"
echo "checkpoint=$CKPT_DIR/$CKPT_NAME"
echo "output=$OUTPUT_DIR"
echo "frame_stride=$FRAME_STRIDE noises=8"
launch_command=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/run_linglong_action_entropy.py \
  --checkpoint-dir "$CKPT_DIR" \
  --checkpoint-name "$CKPT_NAME" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --frame-stride "$FRAME_STRIDE" \
  --noises 8 \
  "$@")
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch_command[@]}"
  echo
  exit 0
fi
exec "${launch_command[@]}"
