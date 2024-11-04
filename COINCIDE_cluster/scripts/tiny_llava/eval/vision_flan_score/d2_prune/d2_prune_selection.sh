#!/bin/bash
BASE_PATH="/base_path"
SAMPLE_RATIO=0.167

# Start time
start_time=$(date +%s)

python tinyllava/eval/score/d2_prune/d2_prune.py \
        --score-path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/aum_scores.npy \
        --embed-path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/avg_llava_embed.npy \
        --output-indices-path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/d2_prune_indices_0.042.npy \
        --n-neighbors 5 \
        --gamma 0.4 \
        --fraction $SAMPLE_RATIO \

# End time
end_time=$(date +%s)
# Calculate execution time
execution_time=$((end_time - start_time))
minutes=$((execution_time/60))
echo "Total execution time: ${minutes} minutes"