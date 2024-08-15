#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa_science \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/scienceqa/images/test \
    --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ${BASE_PATH}/COINCIDE_train/playground/data/eval/scienceqa \
    --result-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --output-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ${BASE_PATH}/COINCIDE_train/playground/data/eval/scienceqa/answers/${CKPT}_result.json
