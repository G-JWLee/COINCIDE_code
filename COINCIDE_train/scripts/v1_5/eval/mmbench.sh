#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
SPLIT="mmbench_dev_20230712"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ${BASE_PATH}/COINCIDE/playground/data/eval/mmbench/answers/$SPLIT/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${BASE_PATH}/COINCIDE/playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${BASE_PATH}/COINCIDE/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ${BASE_PATH}/COINCIDE/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ${BASE_PATH}/COINCIDE/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
