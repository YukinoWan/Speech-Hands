#!/bin/bash

python3 ./scripts/qwen_omni_merge.py save_full \
  --base_model_path="Qwen/Qwen2.5-Omni-7B" \
  --saved_thinker_path="./saves/qwen2_omni-7b/full/${1}-rag-with-audio-sharegpt-${2}/" \
  --save_path="./target_dir/${1}-rag-with-audio-sharegpt-ckpt-${2}"

cp ./target_dir/ami-whisper-5-best-with-audio-sharegpt-ckpt-full/spk_dict.pt ./target_dir/${1}-rag-with-audio-sharegpt-ckpt-${2}/
