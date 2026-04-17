#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/preliminary/ger_no_rag/qwen2_5omni_full_sft_tedlium_with_audio.yaml
bash ./merge_full_ger.sh "tedlium" "parakeet_v2_ger"

bash ./scripts/eval/eval_ger.sh "tedlium_test" "parakeet_v2_ger"
bash ./scripts/eval/eval_ger.sh "tedlium_train" "parakeet_v2_ger"
