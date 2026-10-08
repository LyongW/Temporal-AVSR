#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/pretrained_models/temporal_avsr/checkpoint.pt}
OUT_PATH=${OUT_PATH:-$PROJECT_ROOT/results/clean}
SUBSET=${SUBSET:-test}
if ! configuration_only "$@"; then
    for file in "$MODEL_PATH" "$AVHUBERT_PATH" "$SR_PREDICTOR_PATH" "$NOISE_WAV" \
        "$DATA_DIR/$SUBSET.tsv" "$LABEL_DIR/$SUBSET.wrd"; do
        require_file "$file"
    done
fi
exec python -B "$PROJECT_ROOT/src/eval.py" \
    --config-dir "$PROJECT_ROOT/src/conf" --config-name s2s_decode \
    "common.user_dir=$PROJECT_ROOT/src" "dataset.gen_subset=$SUBSET" \
    "common_eval.path=$MODEL_PATH" "common_eval.results_path=$OUT_PATH" \
    "hydra.run.dir=$OUT_PATH/hydra" \
    "override.data=$DATA_DIR" "override.label_dir=$LABEL_DIR" \
    "override.modalities=$MODALITIES" "override.llm_path=$LLM_PATH" \
    "override.w2v_path=$AVHUBERT_PATH" "override.sr_predictor_path=$SR_PREDICTOR_PATH" \
    "override.whisper_path=$WHISPER_PATH" "override.qformer_path=$QFORMER_PATH" \
    "override.noise_wav=$NOISE_WAV" "override.noise_prob=${NOISE_PROB:-0}" \
    "override.noise_snr=${SNR:-0}" generation.beam=5 generation.temperature=0.3 "$@"
