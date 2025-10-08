#!/bin/bash

#SBATCH --job-name=qwen2_5omni_lora_sharegpt_tedlium-whisper-5-best-with-audio
#SBATCH --output=log/qwen2_5omni_lora_sharegpt_tedlium-whisper-5-best-with-audio
#SBATCH --error=log/qwen2_5omni_lora_sharegpt_tedlium-whisper-5-best-with-audio
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --partition=grizzly,polar,polar3,polar4
#SBATCH --account=nvr_taiwan_rvos
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --overcommit

source /home/zwan/.cache/conda/etc/profile.d/conda.sh
conda activate factory

#default_model="/lustre/fsw/portfolios/nvr/users/hanrongy/checkpoints/xvila/xvila-nvila_8b-video_audBoostv2_s3_n8_bs2048_ga8_mstep-1_j20250414/model"
#default_output_dir="./eval_output"
#default_log_dir="./eval_log"
cd ~/.cache/LLaMA-Factory
#llamafactory-cli train examples/train_lora/qwen2_5omni_lora_sft.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen2_5omni_lora_sft_tedlium_with_audio.yaml
