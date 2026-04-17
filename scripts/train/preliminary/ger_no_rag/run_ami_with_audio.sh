#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/preliminary/ger_no_rag/qwen2_5omni_full_sft_ami_with_audio.yaml
bash ./merge_full_ger.sh "ami" "parakeet_v2_ger"

bash ./scripts/eval/eval_ger.sh "ami_test" "parakeet_v2_ger"
bash ./scripts/eval/eval_ger.sh "ami_train" "parakeet_v2_ger"
