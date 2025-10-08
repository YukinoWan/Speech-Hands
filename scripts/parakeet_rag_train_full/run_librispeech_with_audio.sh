#!/bin/bash

#SBATCH --job-name=qwen2_5omni_full_sharegpt_librispeech-rag-with-audio-v3
#SBATCH --output=parakeet_log/qwen2_5omni_full_sharegpt_librispeech-rag-with-audio
#SBATCH --error=parakeet_log/qwen2_5omni_full_sharegpt_librispeech-rag-with-audio
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --partition=p2,p3
#SBATCH --account=zhenwan.nlp
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --overcommit

source /mnt/home/zhenwan.nlp/miniconda3/etc/profile.d/conda.sh
conda activate factory

#default_model="/lustre/fsw/portfolios/nvr/users/hanrongy/checkpoints/xvila/xvila-nvila_8b-video_audBoostv2_s3_n8_bs2048_ga8_mstep-1_j20250414/model"
#default_output_dir="./eval_output"
#default_log_dir="./eval_log"
cd ~/LLaMA-Factory-Nbest
#llamafactory-cli train examples/train_lora/qwen2_5omni_lora_sft.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/parakeet_rag_train_full/qwen2_5omni_full_sft_librispeech_with_audio_rag.yaml
bash ./merge_full_rag.sh "librispeech" "parakeet-v3"

sbatch ./scripts/parakeet_eval/eval_rag.sh "librispeech_test_clean" "parakeet-v3"
sbatch ./scripts/parakeet_eval/eval_rag.sh "librispeech_test_other" "parakeet-v3"