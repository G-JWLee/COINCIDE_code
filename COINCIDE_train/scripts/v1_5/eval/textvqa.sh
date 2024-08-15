#!/bin/bash

CKPT=$1
BASE="vicuna-7b-v1.5"
BASE_PATH="/base_path"

python -m llava.eval.model_vqa_loader \
    --model-path ${BASE_PATH}/checkpoints/$CKPT \
    --model-base ${BASE_PATH}/checkpoints/$BASE \
    --question-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${BASE_PATH}/COINCIDE_train/playground/data/eval/textvqa/train_images \
    --answers-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/textvqa/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${BASE_PATH}/COINCIDE_train/playground/data/eval/textvqa/answers/${CKPT}.jsonl
