#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/preliminary/whisper_rag_2way/qwen2_5omni_full_sft_voxpopuli_with_audio_rag.yaml
