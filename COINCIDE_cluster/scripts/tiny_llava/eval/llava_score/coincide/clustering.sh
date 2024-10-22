#!/bin/bash

BASE_PATH="/base_path"
SAMPLE_RATIO=0.2
TEMP=0.1

python tinyllava/eval/score/coincide/compute_centroids.py \
        --sim_metric cosine \
        --Kmeans_with_cos_dist \
        --emb_memory_loc ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/tan_act_37111519_msa.npy \
        --save_folder ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/10000_msa_save_folder \
        --ncentroids 10000 \
        --niter 50 \
        --seed 1234 \


python tinyllava/eval/score/coincide/cluster_transferability.py \
        --centroid_embed_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/10000_msa_save_folder/kmeans_centroids.npy \
        --transferability_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/10000_msa_save_folder/transfer.npy \
        --k 4 \
        --knn_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/10000_msa_save_folder/knn \


python tinyllava/eval/score/coincide/cluster_wise_prune.py \
        --embedding_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/tan_act_37111519_msa.npy \
        --cluster_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/10000_msa_save_folder/nearest_cent.npy
        --transfer_path ${BASE_PATH}/COINCIDE_train/playground/data/TinyLLaVA-Instruction/10000_msa_save_folder/transfer.npy \
        --fraction $SAMPLE_RATIO \
        --temp $TEMP \
