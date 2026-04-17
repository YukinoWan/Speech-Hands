#!/bin/bash

#FORCE_TORCHRUN=1 llamafactory-cli train examples/parakeet_rag_train_full/qwen2_5omni_full_sft_spgispeech_with_audio_rag.yaml
bash ./merge_full_rag.sh "spgispeech" "parakeet-ger_rag"

bash ./scripts/parakeet_eval/eval_rag.sh "spgispeech_test" "parakeet-ger_rag"
