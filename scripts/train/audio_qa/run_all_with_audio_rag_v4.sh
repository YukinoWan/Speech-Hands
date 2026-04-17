#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/audio_qa/qwen2_5omni_full_sft_2025_with_audio_rag_v4.yaml
bash merge_full.sh "v4"

CUDA_VISIBLE_DEVICES=0 python test_qwen_2025.py "v4"
