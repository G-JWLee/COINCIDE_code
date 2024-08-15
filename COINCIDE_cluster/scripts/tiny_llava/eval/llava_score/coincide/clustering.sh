#!/bin/bash

BASE_PATH="/base_path"

python tinyllava/eval/score/coincide/compute_centroids.py \
        --sim_metric cosine \
        --Kmeans_with_cos_dist \
        --emb_memory_loc ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/tan_act_37111519_msa.npy \
        --save_folder ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/tan_act_37111519_msa_save_folder \
        --ncentroids 10000 \
        --niter 50 \
        --seed 1234 \


python tinyllava/eval/score/coincide/cluster_transferability.py \
        --centroid_embed_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/tan_act_37111519_msa_save_folder/kmeans_centroids.npy \
        --output_indices_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/tan_act_37111519_msa_save_folder/transfer.npy \
        --k 4 \
        --knn_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/tan_act_37111519_msa_save_folder/knn \
