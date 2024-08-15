#!/bin/bash

BASE_PATH="/base_path"

# Start time
start_time=$(date +%s)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="vison_flan_tinyllava_v100_2.0b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python tinyllava/eval/score/coincide/score_task_embed_tan.py \
        --model_path ${BASE_PATH}/checkpoints/$CKPT \
        --data_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/annotation_191-task_1k.json \
        --image_folder ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/images_191task_1k \
        --score_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --is_multimodal \
        --batch_size 8 \
        --layer_list 3 7 11 15 19 \
        --version phi &
done

wait

python tinyllava/eval/score/merge_values.py \
        --score_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/tan_act_37111519_msa \

wait

# End time
end_time=$(date +%s)
# Calculate execution time
execution_time=$((end_time - start_time))
minutes=$((execution_time/60))
echo "Total execution time: ${minutes} minutes"
