# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
import pickle
import numpy as np
import json

IMAGE_NAME_INDEX = 0

def extract_pruned_data(
    sorted_clusters_path,
    semdedup_pruning_tables_path,
    eps_list,
    num_clusters,
    output_indices_path,
    target_length,
    retreive_kept_samples=True,
):

    target_close = []
    for eps in eps_list:

        ## -- list of paths to the examples we want to keep/remove.
        example_paths = []

        for cluster_id in tqdm(range(0, num_clusters)):

            cluster_i = np.load(
                os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
            )
            with open(
                f"{semdedup_pruning_tables_path}/cluster_{cluster_id}.pkl", "rb"
            ) as file:
                semdedup_pruning_tables = pickle.load(file)

            ## -- See which examples to keep/remove from this cluster.
            ## -- Use retreive_kept_samples=True when kept dataset size <= 50%. This will return a smaller output text file,
            ## -- semdedup_pruning_tables contain True values for the examples to be removed.
            images_to_keep_or_remove = semdedup_pruning_tables[f"eps={eps}"][
                semdedup_pruning_tables[f"eps={eps}"] == (not retreive_kept_samples)
            ].index.to_numpy()
            if "indices" in semdedup_pruning_tables.columns:
                cluster_i = cluster_i[semdedup_pruning_tables["indices"]]
            ## -- retrieve only the examples we want and add to the list.
            dedup_cluster = cluster_i[images_to_keep_or_remove]
            example_paths += dedup_cluster[:, IMAGE_NAME_INDEX].astype("int").tolist()

        if (abs(len(example_paths) - target_length) < abs(len(target_close) - target_length)) and (len(example_paths) - target_length >= 0):
            target_close = example_paths

    assert len(target_close) != 0

    num_diff = len(target_close) - target_length
    print(f"Selected data - Target length: {num_diff}")

    selected_indices = np.array(target_close)
    selected_indices = selected_indices[:target_length]
    np.save(output_indices_path, selected_indices)

    return
