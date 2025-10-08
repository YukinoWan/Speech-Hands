#!/bin/bash

#SBATCH --job-name=qwen2_5omni_full_sharegpt_gigaspeech-ger-with-audio
#SBATCH --output=parakeet_log/qwen2_5omni_full_sharegpt_gigaspeech-ger-with-audio
#SBATCH --error=parakeet_log/qwen2_5omni_full_sharegpt_gigaspeech-ger-with-audio
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --partition=p2,p3
#SBATCH --account=zhenwan.nlp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --overcommit

source /mnt/home/zhenwan.nlp/miniconda3/etc/profile.d/conda.sh
conda activate factory

#default_model="/lustre/fsw/portfolios/nvr/users/hanrongy/checkpoints/xvila/xvila-nvila_8b-video_audBoostv2_s3_n8_bs2048_ga8_mstep-1_j20250414/model"
#default_output_dir="./eval_output"
#default_log_dir="./eval_log"
cd ~/LLaMA-Factory-Nbest
#llamafactory-cli train examples/train_lora/qwen2_5omni_lora_sft.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/parakeet_ger_train_full/qwen2_5omni_full_sft_gigaspeech_with_audio.yaml
bash ./merge_full_ger.sh "gigaspeech" "parakeet_v2_ger"

sbatch ./scripts/parakeet_eval/eval_ger.sh "gigaspeech_test" "parakeet_v2_ger"
sbatch ./scripts/parakeet_eval/eval_ger.sh "gigaspeech_train" "parakeet_v2_ger"