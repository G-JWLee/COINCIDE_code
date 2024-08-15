#!/bin/bash

if [ "$1" == "" ]; then
  PRUNE_RATIO=0.2
else
  PRUNE_RATIO=$1
fi

BASE_PATH="/base_path"

deepspeed llava/train/train_xformers.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ${BASE_PATH}/COINCIDE_train/scripts/zero3.json \
    --model_name_or_path ${BASE_PATH}/checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${BASE_PATH}/COINCIDE_train/playground/data/llava_v1_5_mix665k.json \
    --image_folder ${BASE_PATH}/COINCIDE_train/playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${BASE_PATH}/checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir ${BASE_PATH}/checkpoints/llava_lora_random_prune_${PRUNE_RATIO}_v1.5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --fp16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name llava_lora_random_prune_${PRUNE_RATIO}_v1.5 \
    --prune_indices ${BASE_PATH}/COINCIDE_train/playground/data/LLaVA-Instruction/random_indices.npy \
    --prune_p $PRUNE_RATIO \
    --prune_subset "bottom" \
