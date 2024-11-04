#!/bin/bash
BASE_PATH="/base_path"
CKPT="vison_flan_tinyllava_v100_2.0b"

# Start time
start_time=$(date +%s)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python tinyllava/eval/score/el2n/score_el2n.py \
        --model_path ${BASE_PATH}/checkpoints/$CKPT \
        --data_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/annotation_191-task_1k.json \
        --image_folder ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/images_191task_1k \
        --score_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/el2n_scores \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --is_multimodal \
        --batch_size 4 \
        --version phi &
done

wait

python tinyllava/eval/score/merge_values.py \
        --score_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/el2n_scores \

python tinyllava/eval/score/sort_values.py \
    --score_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/el2n_scores.npy \
    --save_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/el2n_indices.npy \

# End time
end_time=$(date +%s)

# Calculate execution time
execution_time=$((end_time - start_time))
minutes=$((execution_time/60))

echo "Total execution time: ${minutes} minutes"
