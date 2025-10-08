#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=eval_log/qwen2_5omni_full_sharegpt_voxpopuli-whisper-5-best-with-audio-v2-log
#SBATCH --error=eval_log/qwen2_5omni_full_sharegpt_voxpopuli-whisper-5-best-with-audio-v2-err
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --partition=p2
#SBATCH --account=zhenwan.nlp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --overcommit

source /mnt/home/zhenwan.nlp/miniconda3/etc/profile.d/conda.sh
conda activate qwen_eval

#default_model="/lustre/fsw/portfolios/nvr/users/hanrongy/checkpoints/xvila/xvila-nvila_8b-video_audBoostv2_s3_n8_bs2048_ga8_mstep-1_j20250414/model"
#default_output_dir="./eval_output"
#default_log_dir="./eval_log"
cd ~/LLaMA-Factory-Nbest
#llamafactory-cli train examples/train_lora/qwen2_5omni_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 python test_qwen_ger.py "voxpopuli" "full-v2" "False"
