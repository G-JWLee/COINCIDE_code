import argparse
import random
import numpy as np
import os
import logging
from tinyllava.eval.score.SemDeDup.clustering.clustering import compute_centroids
from tinyllava.eval.score.SemDeDup.clustering.sort_clusters import assign_and_sort_clusters
from tinyllava.eval.score.SemDeDup.execute_semdedup import execute_semdedup
from tinyllava.eval.score.SemDeDup.extract_dedup_data import extract_pruned_data

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_metric", type=str, default="cosine")
    parser.add_argument("--keep_hard", action='store_true')
    parser.add_argument("--Kmeans_with_cos_dist", action='store_true')
    parser.add_argument("--emb_memory_loc", type=str, default="emb.npy")
    parser.add_argument("--sorted_clusters_file_loc", type=str, default="./sorted_clusters")
    parser.add_argument("--save_folder", type=str, default="./save_folder")
    parser.add_argument("--output_indices_path", type=str, default="./selected_indices.json")
    parser.add_argument("--ncentroids", type=int, default=500)  # proportional to dataset size
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prune_p", type=float, default=.0)
    parser.add_argument("--eps_list", type=float, nargs='+', default=[0.48, 0.47, 0.46, 0.45, 0.44])

    args = parser.parse_args()

    ## -- Fix the seed
    SEED = args.seed
    random.seed(SEED)

    emb_memory = np.load(args.emb_memory_loc)
    dataset_size, emb_size = emb_memory.shape
    # Normalize since SemDeDup uses Spherical Kmeans clustering with normalized embeddings, referring to paper, even in language modality with OPT model.
    emb_memory = emb_memory / np.linalg.norm(emb_memory, axis=-1, keepdims=True)

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

    indices_list = np.array(list(range(dataset_size)))
    assign_and_sort_clusters(
        data=emb_memory,
        paths_list=indices_list,
        sim_metric=args.sim_metric,
        keep_hard=args.keep_hard,
        kmeans_with_cos_dist=args.Kmeans_with_cos_dist,
        save_folder=args.save_folder,
        sorted_clusters_file_loc=args.sorted_clusters_file_loc,
        cluster_ids=range(0, args.ncentroids),
        logger=logger,
    )

    execute_semdedup(
        embs=emb_memory,
        cluster_ids=range(0, args.ncentroids),
        save_loc=args.save_folder,
        sorted_clusters_path=args.sorted_clusters_file_loc,
        eps_list=args.eps_list,
        which_to_keep= "hard" if args.keep_hard else "easy",
    )

    target_length = int(args.prune_p * dataset_size)
    extract_pruned_data(
        sorted_clusters_path=args.sorted_clusters_file_loc,
        semdedup_pruning_tables_path=os.path.join(args.save_folder, "dataframes"),
        eps_list=args.eps_list,
        num_clusters=args.ncentroids,
        output_indices_path=args.output_indices_path,
        target_length=target_length,
    )
