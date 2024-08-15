#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet/images \
    --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --dst ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet/results/${CKPT}.json

python playground/data/eval/mm-vet/MM-Vet/mm-vet_evaluator.py \
    --mmvet_path ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet \
    --result_file ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet/results/${CKPT}.json \
    --result_path ${BASE_PATH}/COINCIDE_train/playground/data/eval/mm-vet
