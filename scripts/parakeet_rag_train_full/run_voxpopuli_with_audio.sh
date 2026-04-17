#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train examples/parakeet_rag_train_full/qwen2_5omni_full_sft_voxpopuli_with_audio_rag.yaml
bash ./merge_full_rag.sh "voxpopuli" "parakeet-v3"

bash ./scripts/parakeet_eval/eval_rag.sh "voxpopuli_test" "parakeet-v3"
