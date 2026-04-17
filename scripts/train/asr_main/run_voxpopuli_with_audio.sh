#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/asr_main/qwen2_5omni_full_sft_voxpopuli_with_audio_rag.yaml
bash ./merge_full_rag.sh "voxpopuli" "parakeet-v3"

bash ./scripts/eval/eval_rag.sh "voxpopuli_test" "parakeet-v3"
