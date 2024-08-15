#!/bin/bash

BASE_PATH="/base_path"

python tinyllava/eval/score/coincide/compute_centroids.py \
        --sim_metric cosine \
        --Kmeans_with_cos_dist \
        --emb_memory_loc ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/tan_act_37111519_msa.npy \
        --save_folder ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/2500_save_folder \
        --ncentroids 2500 \
        --niter 50 \
        --seed 1234 \


python tinyllava/eval/score/coincide/cluster_transferability.py \
        --centroid_embed_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/2500_save_folder/kmeans_centroids.npy \
        --output_indices_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/2500_save_folder/transfer_lang.npy \
        --k 4 \
        --knn_path ${BASE_PATH}/COINCIDE_train/playground/data/vision-flan_191-task_1k/2500_save_folder/knn \
