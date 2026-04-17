#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/audio_qa/qwen2_5omni_full_sft_2025_with_audio_rag_v6.yaml
bash merge_full.sh "1best-v6"

CUDA_VISIBLE_DEVICES=0 python test_qwen_2025.py "1best-v6"
