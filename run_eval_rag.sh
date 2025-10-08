#!/bin/bash

JOB_NAME="eval"
OUTPUT="parakeet_5_best_eval_log/qwen2_5omni_full_${1}_test_with_audio_rag_ckpt-v1-log"
ERROR="parakeet_5_best_eval_log/qwen2_5omni_full_${1}_test_with_audio_rag_ckpt-v1-err"
PARTITION="p2"
MEM="200G"
TIME="08:00:00"
GPU=1
CPUS=8

# 调用 sbatch 并覆盖部分参数
sbatch \
    --job-name=${JOB_NAME} \
    --output=${OUTPUT} \
    --error=${ERROR} \
    --time=${TIME} \
    --mem=${MEM} \
    --partition=${PARTITION} \
    --gres=gpu:${GPU} \
    --cpus-per-task=${CPUS} \
    scripts/eval/eval_rag.sh ${1} ${2}
