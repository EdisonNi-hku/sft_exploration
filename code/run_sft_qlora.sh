#!/bin/bash

MODEL_NAME=$1
DATA_DIR=$2
CACHE_DIR=$3
OUTPUT_DIR=$4
PACKING=$5
TRAIN_ON_SOURCE=${6:-True}
CONTINUE_ON=${7:-Empty}

python sft_qlora.py \
    --model_name_or_path ${MODEL_NAME} \
    --continue_on ${CONTINUE_ON} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 1 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 40 \
    --evaluation_strategy no \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 512 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --packing ${PACKING}\
    --train_on_source ${TRAIN_ON_SOURCE}\
    --remove_unused_columns True \
    --do_train \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset ${DATA_DIR} \
    --source_max_len 2048 \
    --target_max_len 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
