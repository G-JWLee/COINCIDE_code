import argparse
import random
import numpy as np
import os
import logging
from tinyllava.eval.score.SemDeDup.clustering.clustering import compute_centroids

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_metric", type=str, default="cosine")
    parser.add_argument("--Kmeans_with_cos_dist", action='store_true')
    parser.add_argument("--emb_memory_loc", type=str, default="emb.npy")
    parser.add_argument("--save_folder", type=str, default="./save_folder")
    parser.add_argument("--ncentroids", type=int, default=500)  # proportional to dataset size
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    ## -- Fix the seed
    SEED = args.seed
    random.seed(SEED)

    emb_memory = np.load(args.emb_memory_loc)
    dataset_size, emb_size = emb_memory.shape
    # Normalize since SemDeDup uses Spherical Kmeans clustering with normalized embeddings, referring to paper, even in language modality with OPT model.
    # emb_memory = emb_memory / np.linalg.norm(emb_memory, axis=-1, keepdims=True)

    compute_centroids(
        data=emb_memory,
        ncentroids=args.ncentroids,
        niter=args.niter,
        seed=args.seed,
        Kmeans_with_cos_dist=args.Kmeans_with_cos_dist,
        save_folder=args.save_folder,
        logger=logger,
        verbose=True,
    )

