#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/vizwiz/test \
    --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/vizwiz/answers_upload/${CKPT}.json
