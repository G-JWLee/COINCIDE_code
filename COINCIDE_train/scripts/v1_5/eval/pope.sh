#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/pope/val2014 \
    --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/pope/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ${BASE_PATH}/COINCIDE_train/playground/data/eval/pope/coco \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/pope/answers/${CKPT}.jsonl
