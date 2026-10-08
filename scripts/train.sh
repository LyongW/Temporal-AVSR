#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
NGPUS=${NGPUS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT_PATH=${OUT_PATH:-$PROJECT_ROOT/exp/default}
VALID_SUBSET=${VALID_SUBSET:-valid}

if ! configuration_only "$@"; then
    for file in "$AVHUBERT_PATH" "$SR_PREDICTOR_PATH" "$NOISE_WAV" \
        "$DATA_DIR/train.tsv" "$LABEL_DIR/train.wrd" \
        "$DATA_DIR/$VALID_SUBSET.tsv" "$LABEL_DIR/$VALID_SUBSET.wrd"; do
        require_file "$file"
    done
fi

exec python -B -m fairseq_cli.hydra_train \
    --config-dir "$PROJECT_ROOT/src/conf" --config-name temporal-avsr \
    "common.user_dir=$PROJECT_ROOT/src" \
    "hydra.run.dir=$OUT_PATH" \
    "task.data=$DATA_DIR" "task.label_dir=$LABEL_DIR" \
    "task.llm_path=$LLM_PATH" "task.whisper_path=$WHISPER_PATH" \
    "task.modalities=$MODALITIES" \
    "task.noise_wav=$NOISE_WAV" "task.noise_prob=${NOISE_PROB:-0.75}" \
    "model.w2v_path=$AVHUBERT_PATH" "model.llm_path=$LLM_PATH" \
    "model.sr_predictor_path=$SR_PREDICTOR_PATH" \
    "model.whisper_path=$WHISPER_PATH" "model.qformer_path=$QFORMER_PATH" \
    model.llama_embed_dim=3072 model.target_modules=q_proj.k_proj.v_proj.o_proj \
    model.queries_per_sec=3 model.modality_fuse=concat \
    model.lora_rank=16 model.lora_alpha=32 model.use_qformer=true model.use_sr_predictor=true \
    "dataset.valid_subset=$VALID_SUBSET" \
    optimization.update_freq=[1] optimization.lr=[1e-4] \
    "optimization.max_update=${MAX_UPDATES:-60000}" \
    lr_scheduler._name=cosine lr_scheduler.warmup_updates=500 \
    "distributed_training.distributed_world_size=$NGPUS" \
    "distributed_training.nprocs_per_node=$NGPUS" \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true "$@"
