# Codes from https://github.com/BeenKim/MMD-critic/blob/master/mmd.py

import argparse
import numpy as np
import torch

from tqdm import tqdm


def gaussian_kernel(X, Y, sigma):
    X_norm = X.pow(2).sum(1).view(-1, 1)
    Y_norm = Y.pow(2).sum(1).view(1, -1)
    pairwise_dists = X_norm + Y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))

    # pairwise_dists = cdist(X, Y, 'sqeuclidean')
    K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
    return K


def greedy_mmd_selection(K, M):

    n = len(K)

    indices = np.arange(n)
    selected = np.array([], dtype=int)

    K_XX = K.mean()

    for i in range(M):

        candidates = np.setdiff1d(indices, selected)

        temp_select = np.tile(selected, (len(candidates),1))
        temp_select = np.concatenate([temp_select, candidates[:,np.newaxis]], axis=1)  # Assume that each candidate index is selected

        candidates = torch.from_numpy(candidates).cuda()
        temp_select = torch.from_numpy(temp_select).cuda()

        K_XY = K[:, temp_select]
        K_XY = K_XY.mean(dim=(0,2))

        K_YY = K[temp_select[:,:,None], temp_select[:,None,:]]
        K_YY = K_YY.mean(dim=(1,2))

        MMD = K_XX + K_YY - 2 * K_XY

        best_idx = torch.argmin(MMD)
        best_idx = candidates[best_idx]
        selected = np.append(selected, best_idx.cpu().numpy())

    return selected


# Also, we use the last token of LVLM as the embeddings to calculate the distance, as in SemDeDup-language.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)  # N x D matrix to calculate pair-wise distance.
    parser.add_argument("--cluster_path", type=str, required=True)
    parser.add_argument("--transfer_path", type=str, required=True)

    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to retain")
    parser.add_argument("--gamma", type=float, default=1.0, help="Fraction of the dataset to retain")
    parser.add_argument("--temp", type=float, default=1.0)


    parser.add_argument("--output_indices_path", type=str, required=True, help="Path to output directory")

    args = parser.parse_args()

    embeddings = np.load(args.embedding_path)
    cluster_indices = np.load(args.cluster_path)

    clusters = np.unique(cluster_indices)
    num_clusters = len(clusters)
    avg_num_samples = len(embeddings) / num_clusters

    # Calculate the average number of samples per cluster
    target_num_samples = int(args.fraction * len(embeddings))
    remainings = target_num_samples

    selected_indices = []
    count = 0

    K_list = []

    cluster_density = np.zeros(len(clusters), dtype='float64')
    for cluster_idx in tqdm(clusters):
        i_cluster_indices = np.where(cluster_indices == cluster_idx)[0]
        i_cluster_embeddings = embeddings[i_cluster_indices]
        # Note that the float64 and float16 result is highly different, due to the accuracy.
        i_cluster_embeddings = torch.from_numpy(i_cluster_embeddings).cuda().to(torch.float64)

        i_K = gaussian_kernel(i_cluster_embeddings, i_cluster_embeddings, args.gamma)
        density = i_K.mean()

        cluster_density[cluster_idx] = density.item()

        K_list.append(i_K)

    transferability = np.load(args.transfer_path)
    cluster_score = (1 / cluster_density) * transferability

    # Use the density to select the number of samples in each cluster
    ratio = np.exp(cluster_score / args.temp) / np.sum(np.exp(cluster_score / args.temp))
    ratio_sort_indices = np.argsort(ratio)[::-1]  # Sort in descending order
    ratio = ratio[ratio_sort_indices]
    cluster_score = cluster_score[ratio_sort_indices]

    for idx, cluster_idx in enumerate(tqdm(ratio_sort_indices)):

        i_cluster_indices = np.where(cluster_indices == cluster_idx)[0]
        i_K = K_list[cluster_idx]
        i_target_num_samples = round(remainings * ratio[idx])

        if i_target_num_samples > len(i_cluster_indices):
            i_selected_indices = i_cluster_indices
        else:
            i_proto_indices = greedy_mmd_selection(i_K, i_target_num_samples)
            i_selected_indices = i_cluster_indices[i_proto_indices]

        selected_indices.append(i_selected_indices)
        count = len(i_selected_indices)
        # If not sufficient amount of samples were selected, toss it to the next selections.
        # We do this to satisfy select target_num_samples amounts of sample (if not, less than target_num_samples is sampled).
        remainings = remainings - count
        ratio[idx+1:] = np.exp(cluster_score[idx+1:] / args.temp) / np.sum(np.exp(cluster_score[idx+1:] / args.temp))

    selected_indices = np.concatenate(selected_indices)
    np.save(args.output_indices_path, selected_indices)
