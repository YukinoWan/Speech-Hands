#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/audio_qa/qwen2_5omni_full_sft_2025_with_audio_rag_v2.yaml
