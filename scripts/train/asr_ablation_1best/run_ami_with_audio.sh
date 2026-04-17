#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/asr_ablation_1best/qwen2_5omni_full_sft_ami_with_audio_rag.yaml
bash ./merge_full_rag.sh "ami" "parakeet-1best"

bash ./scripts/eval/eval_rag.sh "ami_test" "parakeet-1best"
