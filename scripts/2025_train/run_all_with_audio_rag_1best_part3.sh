#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train examples/2025_train_full/qwen2_5omni_full_sft_2025_with_audio_rag_1best_part3.yaml
bash merge_full.sh "1best-part3"

CUDA_VISIBLE_DEVICES=0 python test_qwen_2025.py "1best-part3"
