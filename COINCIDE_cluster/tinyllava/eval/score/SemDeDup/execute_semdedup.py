import math
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import random
import math
import time
import pprint
from typing import Union, Optional, List


def contains_duplicates(arr):
    return len(np.unique(arr)) != len(arr)


def semdedup(cluster, cluster_reps, device):
    st = time.time()
    ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
    cluster_reps = cluster_reps.to(device=device, dtype=torch.float32)
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    del cluster_reps
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

    ## -- get paths to cluster i images
    image_urls = cluster[:, 0]

    ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
    assert not contains_duplicates(image_urls)

    ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

    ## -- if the max sim between one example and any other example is > 1-eps, remove this example
    M = torch.max(triu_sim_mat, dim=0)[0].cpu()
    print(f"Step time: {time.time() - st}(s)")

    return M


def execute_semdedup(
    embs: Union[np.memmap, np.ndarray],
    cluster_ids=range(5000),
    save_loc: str = "",
    sorted_clusters_path: str = "",
    eps_list: List = [],
    which_to_keep: str = "hard",
):
    os.makedirs(os.path.join(save_loc, "dataframes"), exist_ok=True)
    for cluster_id in tqdm(cluster_ids):
        step_st = time.time()

        df_file_loc = os.path.join(
            save_loc, f"dataframes/cluster_{cluster_id}.pkl"
        )

        if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
            print(f"{df_file_loc} exists, moving on")
            continue

        ## -- load cluster i representations
        cluster_i = np.load(
            os.path.join(
                sorted_clusters_path, f"cluster_{cluster_id}.npy"
            )
        )
        # 1) store cluster size
        cluster_size = cluster_i.shape[0]
        print("cluster_size: ", cluster_size)

        if cluster_size == 1:
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = [0]
            for eps in eps_list:
                ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                points_to_remove_df[f"eps={eps}"] = [False]
            if save_loc != "":
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)
            print("DONE cluster_id ", cluster_id)
            continue

        ## -- By default, we keep hard examples from groups
        clutser_items_indices = list(range(cluster_size))
        ## -- OR: shuffle cluster to keep random example from each group
        if which_to_keep.lower() == "random":
            random.shuffle(clutser_items_indices)
            cluster_i = cluster_i[clutser_items_indices]
        ## -- OR: reverse cluster to keep easy examples
        if which_to_keep.lower() == "easy":
            clutser_items_indices = clutser_items_indices[::-1]
            cluster_i = cluster_i[clutser_items_indices]

        ## -- indices for cluster items in the dataset
        cluster_ids = cluster_i[:, 1].astype("int32")
        cluster_reps = embs[cluster_ids]
        cluster_reps = torch.tensor(cluster_reps)

        device = 'cuda' if torch.cuda.is_available() else  'cpu'
        M = semdedup(cluster_i, cluster_reps, device)

        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = clutser_items_indices

        for eps in eps_list:
            ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
            eps_points_to_remove = M > 1 - eps
            points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

        if save_loc != "":
            ## --save df
            with open(df_file_loc, "wb") as file:
                pickle.dump(points_to_remove_df, file)

    return
