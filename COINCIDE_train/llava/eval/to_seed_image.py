import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--modify-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    seedbench_list = [json.loads(q) for q in open(args.original_file, "r")]
    new_list = []
    count = 0
    for idx in range(len(seedbench_list)):
        if seedbench_list[idx]['image'].startswith('SEED-Bench-video-image'):
            continue
        else:
            new_list.append(seedbench_list[idx])

    mod_file = os.path.expanduser(args.modify_file)
    os.makedirs(os.path.dirname(mod_file), exist_ok=True)
    mod_file = open(args.modify_file, "w")
    for idx in range(len(new_list)):

        mod_file.write(json.dumps(new_list[idx]) + "\n")

    mod_file.close()

