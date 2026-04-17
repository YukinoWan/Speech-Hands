#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen2_5omni_lora_sft_tedlium_no_audio.yaml
