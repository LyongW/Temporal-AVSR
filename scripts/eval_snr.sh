#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODALITIES=${MODALITIES:-'[video,audio]'}
export NOISE_PROB=1
export SNR=${SNR:-0}
export OUT_PATH=${OUT_PATH:-$SCRIPT_DIR/../results/noisy_snr_$SNR}
exec bash "$SCRIPT_DIR/eval.sh" "$@"
