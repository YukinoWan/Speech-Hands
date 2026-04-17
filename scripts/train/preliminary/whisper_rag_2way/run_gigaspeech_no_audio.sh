#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train configs/preliminary/earliest_baseline/qwen2_5omni_full_sft_gigaspeech_no_audio.yaml
