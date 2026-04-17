#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train examples/rag_train_full/qwen2_5omni_full_sft_gigaspeech_with_audio_rag.yaml
