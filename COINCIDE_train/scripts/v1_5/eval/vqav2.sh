#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
BASE="vicuna-7b-v1.5"
SPLIT="llava_vqav2_mscoco_test-dev2015"
BASE_PATH="/base_path"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${BASE_PATH}/checkpoints/$CKPT \
        --model-base ${BASE_PATH}/checkpoints/$BASE \
        --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/vqav2/test2015 \
        --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${BASE_PATH}/COINCIDE_train/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${BASE_PATH}/COINCIDE_train/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT
#
