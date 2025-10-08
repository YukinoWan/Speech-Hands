python3 ./scripts/qwen_omni_merge.py merge_lora \
	--base_model_path="Qwen/Qwen2.5-Omni-7B" \
	--lora_checkpoint_path="./LLaMA-Factory/saves/qwen2_omni-7b/lora/ami-whisper-5-best-with-audio-sharegpt/checkpoint-8000" \
	--save_path="./target_dir/ami-whisper-5-best-with-audio-sharegpt-ckpt-lora"
