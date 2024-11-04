#!/bin/bash
BASE_PATH="/base_path"
CKPT="TinyLLaVA-2.0B"

# Start time
start_time=$(date +%s)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python tinyllava/eval/score/d2_prune/score_aum.py \
        --model_path ${BASE_PATH}/checkpoints/$CKPT \
        --data_path ${BASE_PATH}/COINCIDE_train/playground/data/llava_v1_5_mix665k.json \
        --image_folder ${BASE_PATH}/COINCIDE_train/playground/data \
        --score_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/aum_scores \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --is_multimodal \
        --batch_size 4 \
        --version phi &
done

wait

python tinyllava/eval/score/merge_values.py \
        --score_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/aum_scores \

# End time
end_time=$(date +%s)

# Calculate execution time
execution_time=$((end_time - start_time))
minutes=$((execution_time/60))

echo "Total execution time: ${minutes} minutes"