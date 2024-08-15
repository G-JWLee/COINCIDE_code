#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/MME/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ${BASE_PATH}/COINCIDE_train/playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
