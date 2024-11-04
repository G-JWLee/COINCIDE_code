#!/bin/bash
BASE_PATH="/base_path"
SAMPLE_RATIO=0.167

# Start time
start_time=$(date +%s)

python tinyllava/eval/score/SemDeDup/compute_centroids.py \
        --sim_metric cosine \
        --keep_hard \
        --Kmeans_with_cos_dist \
        --emb_memory_loc ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/llava_embed.npy \
        --sorted_clusters_file_loc ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/semdedup_llava_sorted_clusters \
        --save_folder ${BASE_PATH}/COINCIDE_trainplayground/data/vision-flan_191-task_1k/semdedup_llava_save_folder \
        --output_indices_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/semdedup_indices_${SAMPLE_RATIO}.npy \
        --ncentroids 3000 \
        --niter 50 \
        --seed 1234 \
        --prune_p $SAMPLE_RATIO \
        --eps_list 0.305 0.31 0.315 \

# For different SAMPLE_RATIO, note that eps_list should be manually found

# End time
end_time=$(date +%s)
# Calculate execution time
execution_time=$((end_time - start_time))
minutes=$((execution_time/60))
echo "Total execution time: ${minutes} minutes"