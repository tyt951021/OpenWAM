#!/usr/bin/env bash
# Three workers x eight GPUs. Entropy for the 179 sheet episodes only.
# Same fittings step-18185 probe and stride 2 as the existing 721 npz files.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$REPO_ROOT/scripts/activate_vepfs_openwam.sh"
# activate points at another checkout. The entropy script is in this one.
export OPENWAM_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

# One worker uses NPROC GPUs. Several workers are discovered from the Volcano
# host list; a single machine with no host list stays at one node.
NPROC="${NPROC:-8}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-24}"
FRAME_STRIDE="${FRAME_STRIDE:-2}"
PYTHON_BIN="${PYTHON_BIN:-/vePFS/tangyuting/envs/openwam/bin/python}"
MASTER_PORT="${MASTER_PORT:-29500}"
CKPT_DIR="${CKPT_DIR:-/vePFS/tangyuting/OpenWAM/outputs/linglong_0923_1010_1012_1017_fittings_eef_gtstate_8gpu_bs4_official_finetune/2026-10-19_15-10-08}"
CKPT_NAME="${CKPT_NAME:-checkpoint_step_18185.safetensors}"
DATASET_DIR="${DATASET_DIR:-/vePFS/data/linglong/linglong_20261019_20261020_openwam/linglong_openwam_train_15hz}"
OUTPUT_DIR="${OUTPUT_DIR:-$OPENWAM_REPO_ROOT/outputs/action_entropy_20261019_20261020_openwam_train_step18185_stride${FRAME_STRIDE}}"

if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ || ! "$FRAME_STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NPROC and FRAME_STRIDE must be positive integers" >&2
  exit 2
fi
if [[ -n "$EXPECTED_WORLD_SIZE" && ! "$EXPECTED_WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXPECTED_WORLD_SIZE must be a positive integer" >&2
  exit 2
fi

role="${MLP_ROLE:-worker}"
upper_role="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
hosts_var="MLP_${upper_role}_ALL_HOSTS"
if [[ ! "$hosts_var" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
  echo "ERROR: invalid MLP_ROLE" >&2
  exit 2
fi
ALL_HOSTS_STR="${!hosts_var:-${MLP_WORKER_ALL_HOSTS:-}}"
ALL_HOSTS_STR="${ALL_HOSTS_STR//,/ }"
ALL_HOSTS_STR="${ALL_HOSTS_STR//$'\n'/ }"
if [[ -z "$ALL_HOSTS_STR" ]]; then
  if [[ "${NNODES:-1}" == 1 ]]; then
    ALL_HOSTS_STR="$(hostname)"
  else
    echo "ERROR: cannot find worker host list; set NNODES=1 for a single machine" >&2
    exit 2
  fi
fi
read -r -a hosts <<< "$ALL_HOSTS_STR"
if (( ${#hosts[@]} == 0 )); then
  echo "ERROR: worker host list is empty" >&2
  exit 2
fi
if [[ -n "${NNODES:-}" && "$NNODES" != "${#hosts[@]}" ]]; then
  echo "ERROR: NNODES=$NNODES disagrees with host count ${#hosts[@]}" >&2
  exit 2
fi
NNODES="${#hosts[@]}"
if [[ -z "${NODE_RANK:-}" ]]; then
  THIS_HOST="$(hostname)"
  THIS_IPS="$(hostname -I 2>/dev/null || true)"
  for i in "${!hosts[@]}"; do
    if [[ "${hosts[$i]}" == "$THIS_HOST" || " $THIS_IPS " == *" ${hosts[$i]} "* ]]; then
      NODE_RANK="$i"
      break
    fi
  done
  if [[ -z "${NODE_RANK:-}" && "$THIS_HOST" =~ worker-([0-9]+)$ ]]; then
    NODE_RANK="$((10#${BASH_REMATCH[1]}))"
  fi
fi
if [[ "$NNODES" == 1 && -z "${NODE_RANK:-}" ]]; then
  NODE_RANK=0
fi
if [[ ! "${NODE_RANK:-}" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); then
  echo "ERROR: cannot resolve a valid NODE_RANK in [0, $((NNODES-1))]" >&2
  exit 2
fi
WORLD_SIZE="$((NNODES * NPROC))"
if [[ -n "$EXPECTED_WORLD_SIZE" && "$WORLD_SIZE" != "$EXPECTED_WORLD_SIZE" ]]; then
  echo "ERROR: expected WORLD_SIZE=$EXPECTED_WORLD_SIZE, got NNODES=$NNODES NPROC=$NPROC" >&2
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

MASTER_ADDR_INPUT="${MASTER_ADDR:-}"
MASTER_ADDR="${MASTER_ADDR:-${hosts[0]}}"
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" != 1 ]]; then
  echo "Rendezvous inputs: MASTER_ADDR=${MASTER_ADDR_INPUT:-<unset>} first_worker=${hosts[0]} NNODES=$NNODES NODE_RANK=$NODE_RANK"
  MASTER_ADDR="$("$PYTHON_BIN" - "$MASTER_ADDR" "${hosts[0]}" "$NNODES" <<'PYADDR'
import ipaddress
import socket
import sys

errors = []
for candidate in dict.fromkeys(sys.argv[1:3]):
    try:
        addresses = socket.getaddrinfo(candidate, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        addresses.sort(key=lambda item: item[0] != socket.AF_INET)
        for _, _, _, _, sockaddr in addresses:
            address = sockaddr[0]
            ip = ipaddress.ip_address(address)
            if ip.is_unspecified or (int(sys.argv[3]) > 1 and (ip.is_loopback or ip.is_link_local)):
                continue
            if errors:
                print("WARNING: invalid MASTER_ADDR; using first worker address. " + "; ".join(errors), file=sys.stderr)
            print(address)
            sys.exit(0)
        errors.append(f"{candidate}: no usable distributed address")
    except (OSError, ValueError) as error:
        errors.append(f"{candidate}: {error}")
print("ERROR: cannot resolve rendezvous master: " + "; ".join(errors), file=sys.stderr)
print("Use the platform worker host list or the actual worker-0 IP; do not use dry-run worker-0 placeholders.", file=sys.stderr)
sys.exit(2)
PYADDR
)"
fi
export MASTER_ADDR MASTER_PORT

mkdir -p "$OPENWAM_REPO_ROOT/logs/action_entropy_20261019_20261020_openwam" "$OUTPUT_DIR"
log_path="$OPENWAM_REPO_ROOT/logs/action_entropy_20261019_20261020_openwam/launch_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" != 1 ]]; then
  exec > >(tee -a "$log_path") 2>&1
  echo "OpenWAM action-entropy log: $log_path"
fi
echo "NNODES=$NNODES NPROC=$NPROC WORLD_SIZE=$WORLD_SIZE NODE_RANK=$NODE_RANK FRAME_STRIDE=$FRAME_STRIDE"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "checkpoint=$CKPT_DIR/$CKPT_NAME"
echo "dataset=$DATASET_DIR"
echo "output=$OUTPUT_DIR"

cd "$OPENWAM_REPO_ROOT"
launch_command=("$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/run_linglong_action_entropy.py \
  --checkpoint-dir "$CKPT_DIR" \
  --checkpoint-name "$CKPT_NAME" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --frame-stride "$FRAME_STRIDE" \
  "$@")
if [[ "${OPENWAM_LAUNCH_DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${launch_command[@]}"
  printf '\n'
  exit 0
fi
exec "${launch_command[@]}"
