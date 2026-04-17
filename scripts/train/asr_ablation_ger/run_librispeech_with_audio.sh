#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/asr_ablation_ger/qwen2_5omni_full_sft_librispeech_with_audio_rag.yaml
bash ./merge_full_rag.sh "librispeech" "parakeet-ger_rag"

bash ./scripts/eval/eval_rag.sh "librispeech_test_clean" "parakeet-ger_rag"
bash ./scripts/eval/eval_rag.sh "librispeech_test_other" "parakeet-ger_rag"
