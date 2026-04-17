#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5omni_full_sft_spgispeech_no_audio.yaml
