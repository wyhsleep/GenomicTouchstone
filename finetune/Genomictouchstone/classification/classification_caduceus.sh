#!/bin/bash


cd ../../..

model_path=/your/path/to/models
data_path=/your/path/to/data/

echo "Model path is set to: $model_path"
echo "Data path is set to: $data_path"




TASKs='core_promoter_annotation'



declare -A PRETRAINED_PATHS DISPLAY_NAMES MODELS TOKENIZERS MODEL_NAMES D_MODELS LRs BATCH_SIZES CONJOIN_TRAIN_DECODER CONJOIN_TEST RC_AUGS

CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256="caduceus_ph_seqlen_131k_d_model_256"
PRETRAINED_PATHS[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="${model_path}/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
DISPLAY_NAMES[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="caduceus_ph_seqlen_131k_d_model_256"
MODELS[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="caduceus"
TOKENIZERS[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="caduceus"
MODEL_NAMES[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="dna_embedding_caduceus"
BATCH_SIZES[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]=16
D_MODELS[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]=256
CONJOIN_TRAIN_DECODER[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="false"
CONJOIN_TEST[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="true"
RC_AUGS[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="false"
LRs[$CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256]="1e-6 1e-5 1e-4"

CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256="caduceus_ps_seqlen_131k_d_model_256"
PRETRAINED_PATHS[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="${model_path}/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
DISPLAY_NAMES[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="caduceus_ps_seqlen_131k_d_model_256"
MODELS[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="caduceus"
TOKENIZERS[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="caduceus"
MODEL_NAMES[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="dna_embedding_caduceus"
BATCH_SIZES[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]=8
D_MODELS[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]=256
CONJOIN_TRAIN_DECODER[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="true"
CONJOIN_TEST[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="false"
RC_AUGS[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="false"
LRs[$CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256]="1e-6 1e-5 1e-4"


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
    local CONJOIN_TRAIN_DECODER="${CONJOIN_TRAIN_DECODER[$CONFIG_NAME]}"
    local CONJOIN_TEST="${CONJOIN_TEST[$CONFIG_NAME]}"
    local RC_AUG="${RC_AUGS[$CONFIG_NAME]}"
    
    local LR_ARRAY=(${LRs[$CONFIG_NAME]})
    echo "LR_ARRAY: ${LR_ARRAY}"

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
                    dataset.rc_aug="${RC_AUG}" \
                    +dataset.conjoin_train="false" \
                    +dataset.conjoin_test="${CONJOIN_TEST}" \
                    +dataset.tokenizer_path="${PRETRAINED_PATH}" \
                    +dataset.benchmark_name="classification" \
                    model="${MODEL}" \
                    model._name_="${MODEL_NAME}" \
                    model.config.d_model=${D_MODEL} \
                    +model.config.conjoin_test="${CONJOIN_TEST}" \
                    optimizer.lr="${LR}" \
                    trainer.max_epochs=10 \
                    train.pretrained_model_path="${PRETRAINED_PATH}" \
                    trainer.devices=${NUM_DEVICES} \
                    decoder.mode='pool' \
                    +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
                    +decoder.conjoin_test="${CONJOIN_TEST}" \
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

run_model $CONFIG_CADUCEUS_PH_SEQLEN_131K_D_MODEL_256 1
# run_model $CONFIG_CADUCEUS_PS_SEQLEN_131K_D_MODEL_256 5 &

echo "Job finished"
