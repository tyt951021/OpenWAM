#!/usr/bin/env bash
# One worker, eight GPUs. Action entropy on the fittings EEF test split.
# Same proxy checkpoint, 8 noises, and stride-2 sampling as the train entropy
# used by DemoSpeedup. Test episode indices restart at 0, so this writes a
# separate directory and does not touch the train npz files.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$REPO_ROOT/scripts/activate_vepfs_openwam.sh"
# The shared activate script points at another checkout. This recipe needs the
# entropy script in the checkout that contains this launcher.
export OPENWAM_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"
DATASET_DIR=/vePFS/data/linglong/linglong_0923_1010_1012_1017_fittings_eef_gtstate_portable/data/linglong_0923_1010_1012_1017_fittings_eef_gtstate_test_15hz
CKPT_DIR=/vePFS/tangyuting/OpenWAM/outputs/linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu_bs4_official_finetune/2026-10-19_15-10-08
CKPT_NAME=checkpoint_step_18185.safetensors
FRAME_STRIDE=2
OUTPUT_DIR="$OPENWAM_REPO_ROOT/outputs/action_entropy_fittings_eef_step18185_stride2_test"
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
  mkdir -p "$OPENWAM_REPO_ROOT/logs/action_entropy_fittings_test" "$OUTPUT_DIR"
  log_path="$OPENWAM_REPO_ROOT/logs/action_entropy_fittings_test/launch_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
  exec > >(tee -a "$log_path") 2>&1
  echo "OpenWAM test action-entropy log: $log_path"
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
