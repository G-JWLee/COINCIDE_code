#!/bin/bash
BASE_PATH="/base_path"
CKPT="TinyLLaVA-2.0B"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python tinyllava/eval/score/d2_prune/score_llava_embed.py \
        --model_path ${BASE_PATH}/checkpoints/$CKPT \
        --emb_memory_loc ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/avg_llava_embed \
        --data_path ${BASE_PATH}/COINCIDE_train/playground/data/llava_v1_5_mix665k.json \
        --image_folder ${BASE_PATH}/COINCIDE_train/playground/data \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --is_multimodal \
        --version phi \
        --avg_embed \
        --batch_size 4 &
done

wait

python tinyllava/eval/score/merge_values.py \
        --score_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/avg_llava_embed \
