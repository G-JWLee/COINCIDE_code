# Here, we assume that the embeddings, clip-llava_score is already calculated
# Codes from https://github.com/adymaharana/d2pruning?tab=readme-ov-file

import os
import math
import argparse
import time
import numpy as np
import faiss
import torch
import copy
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--centroid_embed_path", type=str, required=True)
    parser.add_argument("--output_indices_path", type=str, required=True, help="Path to output directory")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--knn_path", type=str, required=True)

    args = parser.parse_args()

    centroid_embed = np.load(args.centroid_embed_path)
    centroid_embed = centroid_embed.reshape(-1, 5, 4096)
    centroid_embed = centroid_embed[:,:,2048:]
    centroid_embed = centroid_embed.reshape(-1, 5*2048)

    cosine_sim = cosine_similarity(centroid_embed, centroid_embed)

    knn_cluster_indices = np.argsort(cosine_sim, axis=-1)[:,::-1][:,:args.k+1]
    knn_cluster_similarity = cosine_sim[np.arange(len(cosine_sim))[:,None], knn_cluster_indices]

    np.save(args.knn_path + '_indices.npy', knn_cluster_indices)
    np.save(args.knn_path + '_similarity.npy', knn_cluster_similarity)

    mask = cosine_sim > 0.9
    cosine_sim[mask] = 0
    transfer = cosine_sim.sum(axis=-1) / (~mask).sum(axis=-1)
    # transfer = cosine_sim.mean(axis=-1)
    np.save(args.output_indices_path, transfer)
