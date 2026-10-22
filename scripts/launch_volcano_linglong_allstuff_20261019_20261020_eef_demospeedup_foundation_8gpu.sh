#!/usr/bin/env bash
# One worker, eight GPUs. DemoSpeedup stride 2/3, trained from Foundation.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$REPO_ROOT/scripts/activate_vepfs_openwam.sh"
# The shared activate script points at another checkout. This recipe needs the
# dataloader in the checkout that contains this script.
export OPENWAM_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"
CONFIG_NAME=train_linglong_allstuff_20261019_20261020_eef_demospeedup_foundation_8gpu
DATASET_DIR=/vePFS/data/linglong/linglong_allstuff_20261019_20261020_eef_train_15hz
ENTROPY_DIR=/vePFS/zhuoyu/OpenWAM/outputs/action_entropy_allstuff_20261019_20261020_eef_step18185_stride2
FOUNDATION="$REPO_ROOT/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"
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
unset OPENWAM_LINGLONG_SOURCE_ROOT OPENWAM_LINGLONG_DATASET_DIR
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/openwam-torch-extensions}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" != 1 ]]; then
  mkdir -p "$OPENWAM_REPO_ROOT/logs/linglong_allstuff_demospeedup_foundation8"
  log_path="$OPENWAM_REPO_ROOT/logs/linglong_allstuff_demospeedup_foundation8/launch_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
  exec > >(tee -a "$log_path") 2>&1
  echo "OpenWAM all-stuff DemoSpeedup foundation launch log: $log_path"
fi
for artifact in manifest.json meta/normalization_stats.npy meta/state_eef_stats.json meta/action_eef_stats.json meta/episodes.jsonl; do
  [[ -s "$DATASET_DIR/$artifact" ]] || { echo "ERROR: missing data asset: $DATASET_DIR/$artifact" >&2; exit 2; }
done
[[ -s "$FOUNDATION/config.yaml" && -s "$FOUNDATION/checkpoint_step_154000.safetensors" ]] || {
  echo "ERROR: missing Foundation checkpoint: $FOUNDATION" >&2
  exit 2
}
[[ -d "$ENTROPY_DIR" ]] || { echo "ERROR: missing entropy directory: $ENTROPY_DIR" >&2; exit 2; }
episode_count=$(grep -c . "$DATASET_DIR/meta/episodes.jsonl")
entropy_count=$(find "$ENTROPY_DIR" -maxdepth 1 -name 'episode_*.npz' | wc -l)
if [[ "$entropy_count" -ne "$episode_count" ]]; then
  echo "ERROR: entropy files $entropy_count != episodes $episode_count in $ENTROPY_DIR" >&2
  exit 2
fi
echo "OpenWAM all-stuff DemoSpeedup foundation entropy files: $entropy_count"
launch_command=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py "--config-name=$CONFIG_NAME" "$@")
echo "OpenWAM DemoSpeedup: Foundation init; precision stride 2; casual stride 3; 5 epochs"
echo "checkpoints: $OPENWAM_REPO_ROOT/outputs/linglong_allstuff_20261019_20261020_eef_demospeedup_foundation_8gpu"
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch_command[@]}"
  echo
  exit 0
fi
exec "${launch_command[@]}"
