#!/usr/bin/env bash
# Shared defaults. Source this file from a launch script.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/fairseq${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data/manifest}
LABEL_DIR=${LABEL_DIR:-$DATA_DIR}
AVHUBERT_PATH=${AVHUBERT_PATH:-$PROJECT_ROOT/pretrained_models/avhubert/large_vox_iter5.pt}
SR_PREDICTOR_PATH=${SR_PREDICTOR_PATH:-$PROJECT_ROOT/pretrained_models/sr_predictor/checkpoint.pt}
LLM_PATH=${LLM_PATH:-meta-llama/Llama-3.2-3B}
WHISPER_PATH=${WHISPER_PATH:-openai/whisper-medium.en}
QFORMER_PATH=${QFORMER_PATH:-bert-large-uncased}
NOISE_WAV=${NOISE_WAV:-$PROJECT_ROOT/noise/babble_noise.wav}
MODALITIES=${MODALITIES:-'[video]'}

configuration_only() {
    for argument in "$@"; do
        case "$argument" in --cfg|--help|-h|--hydra-help) return 0 ;; esac
    done
    return 1
}

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Missing file: %s\nConfigure the corresponding path; see README.md.\n' "$1" >&2
        exit 1
    fi
}
