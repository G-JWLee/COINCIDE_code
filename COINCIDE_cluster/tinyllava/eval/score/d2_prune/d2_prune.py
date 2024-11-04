# Here, we assume that the embeddings, clip-score is already calculated
# Codes from https://github.com/adymaharana/d2pruning?tab=readme-ov-file

import os
import math
import argparse
import time
import numpy as np
import faiss
import torch

from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

def create_faiss_index(train_embeddings, d=768):
    # took 5 minutes
    # took 12 minuts for image+text

    # IVF (Inverted Files) -> each cluster's centroid id is connected with the vectors inside the cluster.
    m = 16  # number of centroid IDs in final compressed vectors
    bits = 8  # number of bits in each centroid
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

    print(train_embeddings.shape)
    # In order to perform efficient similarity searches using the IVFPQ index, it needs to learn the quantization parameters from a set of representative vectors.
    train_embeddings = np.float32(train_embeddings)
    start_time = time.time()
    index.train(train_embeddings)
    print("--- took %s seconds to train ---" % (time.time() - start_time))

    return index

def add_to_index(features, index):

    overall_start_time = time.time()

    features = np.float32(features)
    index.add(features)

    print("--- took %s seconds to add all embeddings ---" % (time.time() - overall_start_time))

    return index


def compute_graph_density(embeddings, n_neighbor, importance_scores):

    # Pairwise_distances makes error since embeddings size is large (4096D), eventhough the # of sampels is ~700k.
    distances = pairwise_distances(embeddings, embeddings)
    num_samples = embeddings.shape[0]

    # distances = np.zeros((num_samples, num_samples), dtype=embeddings.dtype)
    # chunk_size = 5000
    # for i in tqdm(range(0, num_samples, chunk_size)):
    #     embeddings_batch = embeddings[i:i+chunk_size]
    #     distances[i:i+chunk_size, :] = np.sqrt(((embeddings_batch[:,None] - embeddings) ** 2).sum(axis=2))

    epsilon = 0.0000001
    connect = kneighbors_graph(embeddings, n_neighbor, p=2)
    connect = connect.todense()

    neighbors = connect.nonzero()
    inds = zip(neighbors[0], neighbors[1])
    print("%s connected nodes" % len(neighbors[0]))
    # Graph edges are weighted by applying gaussian kernel to manhattan dist.
    # By default, gamma for rbf kernel is equal to 1/n_features but may
    # get better results if gamma is tuned.
    for entry in inds:
        i = entry[0]
        j = entry[1]
        distance = distances[i, j]
        weight_j = np.exp(-distance) * max(importance_scores[j].item(), epsilon)
        weight_i = np.exp(-distance) * max(importance_scores[i].item(), epsilon)
        connect[i, j] = weight_j
        connect[j, i] = weight_i

    graph_density = np.zeros(num_samples)
    for i in np.arange(num_samples):
        graph_density[i] = connect[i, :].sum() + importance_scores[i].item()

    return graph_density, connect, distances

def select_batch(graph_density, connect, distances, gamma: float = 1.0):
    # If a neighbor has already been sampled, reduce the graph density
    # for its direct neighbors to promote diversity.
    batch = []

    num_samples = len(graph_density)

    while len(batch) < num_samples:
        selected = np.argmax(graph_density)
        neighbors = (connect[selected,:] > 0).nonzero()[1]
        graph_density[neighbors] = graph_density[neighbors] - np.exp(-distances[selected, neighbors] * gamma) * graph_density[selected]

        batch.append(selected)
        # If a neighbor has already been sampled, reduce the graph density
        # for its direct neighbors to promote diversity.
        graph_density[list(batch)] = min(graph_density) - 1

    indices = np.array(batch)

    return indices

def initialize_graph(index, sample_scores, embeddings, top_k=5):

    gamma = 1.
    step = 50000
    epsilon = 0.0000001

    total_samples = embeddings.shape[0]
    total_splits = math.ceil(total_samples / step)
    graph_scores = []
    neighbors = []
    distances = []

    for i in tqdm(range(0, total_splits), desc='Iterating over %s splits' % total_splits):

        D_raw, I_raw = index.search(np.float32(embeddings[step * i:step * (i+1)]), k=top_k + 1)
        D_raw = D_raw - np.tile(np.expand_dims(D_raw[:, 0].transpose(), 1), (1, top_k + 1))
        D, I = [], []

        D.append(D_raw[:, 1:top_k + 1])
        I.append(I_raw[:, 1:top_k + 1])

        D = np.concatenate(D, axis=0)
        I = np.concatenate(I, axis=0)  # TODO: Check the range of the I. It should be over 50,000

        distances.append(D)
        dist = np.exp(-D * gamma) * np.maximum(sample_scores[I], np.ones_like(I) * epsilon)

        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=-1)
        scores = sample_scores[step * i:step * (i + 1)]
        multiplier = np.sum(dist, axis=-1)
        scores = multiplier + np.maximum(scores, np.ones_like(scores) * epsilon)

        graph_scores.append(scores)
        neighbors.append(I)

    return np.concatenate(neighbors), np.concatenate(graph_scores), np.concatenate(distances)


def iterative_selection(graph_scores, neighbors, scores, distances, gamma: float = 1.0, fraction: float = 1.0):

    # sort
    num_samples = graph_scores.shape[0]
    num_select = int(fraction * num_samples)
    print(scores.shape, graph_scores.shape)

    print("Min and max of scores are %s and %s" % (np.min(scores), np.max(scores)))
    print("Min and max of updated node scores are %s and %s" % (np.min(graph_scores), np.max(graph_scores)))

    ## Inverse since higher the score, easier the sample.
    # sorted_indices = np.argsort(graph_scores)[::-1]

    # In this setting, we use AUM. Higher the value, easier the sample like CLIP-score.
    sorted_indices = np.argsort(graph_scores)[::-1]

    start_time = time.time()
    counter = 0
    selected = np.zeros(graph_scores.shape[-1])
    print(selected.shape)
    selected_indices = []
    standby = []

    while len(selected_indices) < num_select:

        if len(standby) >= 1 and np.max([graph_scores[standby]]) >= graph_scores[sorted_indices[counter]]:
            idx = np.argmax(graph_scores[standby])
            selected_indices.append(standby[idx])
            graph_scores[neighbors[standby[idx]]] = graph_scores[neighbors[standby[idx]]] - graph_scores[
                standby[idx]] * np.exp(-distances[standby[idx]] * gamma)
            selected[standby[idx]] = 1
            del standby[idx]
            continue
        else:
            if selected[sorted_indices[counter]] == 0:

                # check for more than last selected score and next score and put on standby
                if graph_scores[sorted_indices[counter]] < graph_scores[sorted_indices[counter + 1]]:
                    standby.append(sorted_indices[counter])
                else:
                    selected_indices.append(sorted_indices[counter])
                    graph_scores[neighbors[sorted_indices[counter]]] = graph_scores[neighbors[sorted_indices[counter]]] - \
                                                                    graph_scores[sorted_indices[counter]] * np.exp(
                        -distances[sorted_indices[counter]] * gamma)
                    selected[sorted_indices[counter]] = 1

            counter += 1

        if counter % 50000 == 0:
            print("Selected %s samples" % len(selected_indices))
            print("Length of standby cache: ", len(standby))

    print("-------- Took %s seconds to iterate" % (time.time() - start_time))

    print("Selected %s samples" % len(selected_indices))
    return np.array(selected_indices)

# Also, we use the last token of LVLM as the embeddings to calculate the distance, as in SemDeDup-language.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--score-path", type=str, required=True, help="Path to clip score numpy file")
    parser.add_argument("--embed-path", type=str, required=True, help="Path to clip embedding numpy file")

    parser.add_argument("--output-indices-path", type=str, required=True, help="Path to output directory")
    parser.add_argument("--n-neighbors", type=int, default=1, help="Number of nearest neighbors in D2 Pruning")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for reverse message passing")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to retain")


    args = parser.parse_args()

    # Step 0: Get all scores
    scores = np.load(args.score_path)

    # Step 1: Pool the features
    embeddings = np.load(args.embed_path)

    # Since we cannot compute L2 distance matrices NxN due to memory issue, we use faiss index.
    # Step 2: Add embeddings to index
    index = create_faiss_index(embeddings, embeddings.shape[-1])
    index = add_to_index(embeddings, index)

    # Step 4: Initialize grap
    neighbors, graph_scores, distances = initialize_graph(index, scores, embeddings, args.n_neighbors)

    # Step 5: Iterative selection
    selected_indices = iterative_selection(graph_scores, neighbors, scores, distances, args.gamma, args.fraction)

    np.save(args.output_indices_path, selected_indices)
