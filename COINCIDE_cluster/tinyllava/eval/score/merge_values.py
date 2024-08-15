import argparse
import torch
import numpy as np
import os
import json
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, default=".")
    parser.add_argument("--average", action='store_true')
    args = parser.parse_args()

    score_files = sorted(glob.glob(args.score_path + '_[0-9]*.npy'))

    scores = []
    for score_file in score_files:
        scores.append(np.load(score_file))

    scores = np.concatenate(scores)
    if args.average:
        scores = np.mean(scores, axis=0)

    recover_indices = np.load(args.score_path + '_recover_indices.npy')
    scores = scores[recover_indices]

    np.save(args.score_path + '.npy', scores)

    for score_file in score_files:
        os.remove(score_file)
    os.remove(args.score_path + '_recover_indices.npy')
