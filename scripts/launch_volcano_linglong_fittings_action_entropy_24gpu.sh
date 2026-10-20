#!/usr/bin/env bash
set -euo pipefail
# Three workers x eight GPUs. Keep one frame out of every two.
export NPROC=8
export EXPECTED_WORLD_SIZE=24
export FRAME_STRIDE=2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/launch_volcano_linglong_fittings_action_entropy_8gpu.sh" "$@"
