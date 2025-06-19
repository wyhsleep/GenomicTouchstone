# !/bin/bash

cd ../../..

model_path=/your/path/to/models
data_path=/your/path/to/data/

echo "Model path is set to: $model_path"
echo "Data path is set to: $data_path"

TASKs='core_promoter_annotation'


declare -A PRETRAINED_PATHS DISPLAY_NAMES MODELS TOKENIZERS MODEL_NAMES D_MODELS LRs BATCH_SIZES MAX_LENGTHS RETURN_MASKs CONFIG_PATHS


CONFIG_HYENA_LARGE_1M="hyena_large_1m"
CONFIG_PATHS[$CONFIG_HYENA_LARGE_1M]="${model_path}/hyenadna-large-1m-seqlen/config.json"
PRETRAINED_PATHS[$CONFIG_HYENA_LARGE_1M]="${model_path}/hyenadna-large-1m-seqlen/weights.ckpt"
DISPLAY_NAMES[$CONFIG_HYENA_LARGE_1M]="hyena_large_1m"
MODELS[$CONFIG_HYENA_LARGE_1M]="hyena"
TOKENIZERS[$CONFIG_HYENA_LARGE_1M]="hyena"
MODEL_NAMES[$CONFIG_HYENA_LARGE_1M]="dna_embedding"
LRs[$CONFIG_HYENA_LARGE_1M]="6e-6 6e-5"
BATCH_SIZES[$CONFIG_HYENA_LARGE_1M]=64

CONFIG_HYENA_MEDIUM_160K="hyena_medium_160k"
CONFIG_PATHS[$CONFIG_HYENA_MEDIUM_160K]="${model_path}/hyenadna-medium-160k-seqlen/config.json"
PRETRAINED_PATHS[$CONFIG_HYENA_MEDIUM_160K]="${model_path}/hyenadna-medium-160k-seqlen/weights.ckpt"
DISPLAY_NAMES[$CONFIG_HYENA_MEDIUM_160K]="hyena_medium_160k"
MODELS[$CONFIG_HYENA_MEDIUM_160K]="hyena"
TOKENIZERS[$CONFIG_HYENA_MEDIUM_160K]="hyena"
MODEL_NAMES[$CONFIG_HYENA_MEDIUM_160K]="dna_embedding"
LRs[$CONFIG_HYENA_MEDIUM_160K]="6e-6 6e-5"
BATCH_SIZES[$CONFIG_HYENA_MEDIUM_160K]=32




run_model() {
    local CONFIG_NAME=$1
    local GPU_ID=$2

    local PRETRAINED_PATH="${PRETRAINED_PATHS[$CONFIG_NAME]}"
    local DISPLAY_NAME="${DISPLAY_NAMES[$CONFIG_NAME]}"
    local MODEL="${MODELS[$CONFIG_NAME]}"
    local TOKENIZER="${TOKENIZERS[$CONFIG_NAME]}"
    local MODEL_NAME="${MODEL_NAMES[$CONFIG_NAME]}"
    local D_MODEL="${D_MODELS[$CONFIG_NAME]}"
    local BATCH_SIZE="${BATCH_SIZES[$CONFIG_NAME]}"
    local CONFIG_PATH="${CONFIG_PATHS[$CONFIG_NAME]}"
    local LR_ARRAY=(${LRs[$CONFIG_NAME]})
    
    echo "LRs: ${LR_ARRAY}"

    local GPU_LIST=$(echo "${GPU_ID}" | tr ' ' ',')
    echo "Using GPUs: ${GPU_LIST}"

    local NUM_DEVICES=$(echo "${GPU_LIST}" | tr -cd ',' | wc -c)
    NUM_DEVICES=$((NUM_DEVICES + 1))
    echo "Detected ${NUM_DEVICES} GPUs."

    TASK_ARRAY=(${TASKs})

    for LR in "${LR_ARRAY[@]}"; do
        for seed in $(seq 1 1); do
            for TASK in "${TASK_ARRAY[@]}"; do
                HYDRA_RUN_DIR=${data_path}/${TASK}/finetune/${DISPLAY_NAME}/seed-${seed}_lr-${LR}
                mkdir -p "${HYDRA_RUN_DIR}"
                
                echo "*****************************************************"
                echo "gpu_id: ${GPU_LIST}, seed: ${seed}"
                CUDA_VISIBLE_DEVICES=${GPU_LIST} python -m train \
                    experiment=genomictouchstone/classification \
                    dataset.dataset_name="${TASK}" \
                    dataset.batch_size=${BATCH_SIZE} \
                    dataset.train_val_split_seed=${seed} \
                    dataset.tokenizer_name="${TOKENIZER}" \
                    dataset.dest_path=${data_path} \
                    +dataset.benchmark_name="classification" \
                    +dataset.return_mask=True \
                    model="${MODEL}" \
                    model._name_="${MODEL_NAME}" \
                    +model.config_path="${CONFIG_PATH}" \
                    optimizer.lr="${LR}" \
                    trainer.max_epochs=10 \
                    train.pretrained_model_path="${PRETRAINED_PATH}" \
                    trainer.devices=${NUM_DEVICES} \
                    decoder.mode='pool_masked' \
                    hydra.run.dir="${HYDRA_RUN_DIR}" \
                    wandb.project="fine_tune" \
                    wandb.name="${DISPLAY_NAME}_genomictouchstone_classification_${TASK}_seed-${seed}_lr-${LR}" \
                    wandb.group="genomictouchstone_classification" \
                    +wandb.tags=["seed-${seed}"] \
                    +wandb.tags=["lr-${LR}"] \
                    wandb.job_type="genomictouchstone_classification_${TASK}"
                
                echo "*****************************************************"
            done
        done
    done
}

echo "Starting parallel execution of all configurations"


run_model $CONFIG_HYENA_LARGE_1M 1 
# run_model $CONFIG_HYENA_MEDIUM_160K 2 & 

wait

echo "Job finished"
