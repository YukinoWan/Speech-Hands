#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=qwen3_eval_log/qwen3omni_ami-official-baseline-with-audio-log
#SBATCH --error=qwen3_eval_log/qwen3omni_ami-official-baseline-with-audio-err
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --partition=p2
#SBATCH --account=zhenwan.nlp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --overcommit

source /mnt/home/zhenwan.nlp/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

#default_model="/lustre/fsw/portfolios/nvr/users/hanrongy/checkpoints/xvila/xvila-nvila_8b-video_audBoostv2_s3_n8_bs2048_ga8_mstep-1_j20250414/model"
#default_output_dir="./eval_output"
#default_log_dir="./eval_log"
cd ~/LLaMA-Factory-Nbest
#llamafactory-cli train examples/train_lora/qwen2_5omni_lora_sft.yaml
python test_qwen3_audio_baseline.py "ami_test" ${1}
