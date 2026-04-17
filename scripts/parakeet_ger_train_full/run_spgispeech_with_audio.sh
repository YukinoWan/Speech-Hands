#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train examples/parakeet_ger_train_full/qwen2_5omni_full_sft_spgispeech_with_audio.yaml
bash ./merge_full_ger.sh "spgispeech" "parakeet_v2_ger"

bash ./scripts/parakeet_eval/eval_ger.sh "spgispeech_test" "parakeet_v2_ger"
bash ./scripts/parakeet_eval/eval_ger.sh "spgispeech_train" "parakeet_v2_ger"
