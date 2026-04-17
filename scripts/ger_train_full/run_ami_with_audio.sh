#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5omni_full_sft_ami_with_audio.yaml
