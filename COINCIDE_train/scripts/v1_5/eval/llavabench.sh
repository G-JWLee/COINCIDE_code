#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule ${BASE_PATH}/COINCIDE_train/llava/eval/table/rule.json \
    --answer-list \
        ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/answers/${CKPT}.jsonl \
    --output \
        ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/reviews/${CKPT}.jsonl

python llava/eval/summarize_gpt_review.py -f ${BASE_PATH}/COINCIDE_train/playground/data/eval/llava-bench-in-the-wild/reviews/${CKPT}.jsonl
