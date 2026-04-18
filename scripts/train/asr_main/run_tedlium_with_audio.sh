#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/asr_main/qwen2_5omni_full_sft_tedlium_with_audio_rag.yaml
bash ./merge_full_rag.sh "tedlium" "speech-hands-v3"

bash ./scripts/eval/eval_rag.sh "tedlium_test" "speech-hands-v3"
