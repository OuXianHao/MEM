#!/usr/bin/env bash
set -euo pipefail

# Minimal incremental SFT + LoRA launcher.
# NOTE: if VAL_PARQUET is not provided, TRAIN_PARQUET will be reused for validation.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B}
TRAIN_PARQUET=${TRAIN_PARQUET:-data/incremental_sft_train.parquet}
VAL_PARQUET=${VAL_PARQUET:-$TRAIN_PARQUET}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/mem1_incremental_sft_lora}

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
TARGET_MODULES=${TARGET_MODULES:-[q_proj,v_proj]}

MAX_LENGTH=${MAX_LENGTH:-8196}
BSZ=${BSZ:-64}
MICRO_BSZ=${MICRO_BSZ:-8}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}

PROJECT_NAME=${PROJECT_NAME:-mem1-incremental-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-incremental-sft-lora}
N_GPUS=${N_GPUS:-4}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.fsdp_sft_trainer \
  model.partial_pretrain=$BASE_MODEL \
  data.train_files=$TRAIN_PARQUET \
  data.val_files=$VAL_PARQUET \
  data.tokenized_input=true \
  data.max_length=$MAX_LENGTH \
  data.train_batch_size=$BSZ \
  data.micro_batch_size=$MICRO_BSZ \
  model.lora_rank=$LORA_RANK \
  model.lora_alpha=$LORA_ALPHA \
  model.target_modules=$TARGET_MODULES \
  optim.lr=$LR \
  trainer.total_epochs=$EPOCHS \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.default_hdfs_dir=null \
  trainer.logger=['console']
