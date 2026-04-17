#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/asr_ablation_1best/qwen2_5omni_full_sft_librispeech_with_audio_rag.yaml
bash ./merge_full_rag.sh "librispeech" "parakeet-1best"

bash ./scripts/eval/eval_rag.sh "librispeech_test_clean" "parakeet-1best"
bash ./scripts/eval/eval_rag.sh "librispeech_test_other" "parakeet-1best"
