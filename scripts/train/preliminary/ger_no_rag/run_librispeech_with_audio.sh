#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/preliminary/ger_no_rag/qwen2_5omni_full_sft_librispeech_with_audio.yaml
bash ./merge_full_ger.sh "librispeech" "parakeet_v2_ger"

bash ./scripts/eval/eval_ger.sh "librispeech_test_clean" "parakeet_v2_ger"
bash ./scripts/eval/eval_ger.sh "librispeech_train" "parakeet_v2_ger"
bash ./scripts/eval/eval_ger.sh "librispeech_test_other" "parakeet_v2_ger"
