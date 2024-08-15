import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str, default=".")
    parser.add_argument("--save_path", type=str, default="./temp.json")
    args = parser.parse_args()

    scores = np.load(args.score_path)
    score_indices = np.argsort(scores)

    np.save(args.save_path, score_indices)


