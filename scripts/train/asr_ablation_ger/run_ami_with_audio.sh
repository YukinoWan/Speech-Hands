#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/asr_main/qwen2_5omni_full_sft_ami_with_audio_rag.yaml
bash ./merge_full_rag.sh "ami" "parakeet-ger_rag"

bash ./scripts/eval/eval_rag.sh "ami_test" "parakeet-ger_rag"
