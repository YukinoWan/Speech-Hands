# import soundfile as sf
# from io import BytesIO
# from urllib.request import urlopen
# from qwen_omni_utils import process_vision_info
# from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
# import json
# import os
# from tqdm import tqdm
# from jiwer import wer
# from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sys

import argparse
import csv
import itertools
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger
from huggingface_hub import snapshot_download
from peft import PeftModel
from collections import defaultdict
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation
from llava.mm_utils import process_sounds, process_sound_masks

# normalizer = BasicTextNormalizer()

ckpt = sys.argv[1]
# if "all" in ckpt:
#     model_path = f"./target_dir/all-rag-with-audio-sharegpt-ckpt-{ckpt}"
# else:
#     model_path = f"./target_dir/{task.split('_')[0]}-rag-with-audio-sharegpt-ckpt-{ckpt}"
model_base = "nvidia/audio-flamingo-3"
no_audio = False

dist.init()
devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
torch.cuda.set_device(devices[0])

print(no_audio)
model_path = snapshot_download(model_base)
model_think = os.path.join(model_path, 'stage35')
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2") 
# processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
# from qwen_omni_utils import process_mm_info
model = llava.load(model_path, devices=devices)
# if args.think_mode:
#     model = PeftModel.from_pretrained(
#         model,
#         model_think,
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )
model = model.to("cuda")
# Set up generation config
generation_config = model.default_generation_config
generation_config.max_new_tokens = 1024
# generation_config.num_beams = 5
generation_config.num_return_sequences = 5
generation_config.do_sample = True
generation_config.temperature = 0.8
generation_config.top_p = 0.95
generation_config.top_k = 40
generation_config.repetition_penalty = 1.2
generation_config.length_penalty = 1.0
print(generation_config)
# assert False

if no_audio:
    with open(f"/mnt/home/zhenwan.nlp/2025_DCASE_AudioQA_Official/2025_train_sharegpt.json", "r") as f:
        data = json.load(f)
else:
    with open(f"/mnt/home/zhenwan.nlp/2025_DCASE_AudioQA_Official/2025_train_sharegpt.json", "r") as f:
        data = json.load(f)

output_dir = f"2025_output/with_audio_ckpt-{ckpt}" if not no_audio else f"2025_output/no_audio_ckpt-{ckpt}"
#output_dir = f"output/{task}_ger_baseline_with_audio"
os.makedirs(output_dir, exist_ok=True)

try:
    with open(os.path.join(output_dir, "output.json"), "r") as f:
        exist_data = json.load(f)
except:
    exist_data = []

system_prompt_v1 = "You are a helpful assistant."
system_prompt_v4 = "You are an expert in correcting speech recognition errors. Your task is to improve transcriptions by using both the audio and multiple ASR hypotheses. Always aim to produce accurate and fluent text that closely matches the spoken content. Prioritize correctness and faithfulness over over-correction."
system_prompt_v5 = "You are a speech correction specialist. You listen to audio and compare different ASR hypotheses to find and fix transcription errors. Your goal is to generate the most accurate and natural transcription, using the audio as the ultimate reference."
system_prompt_v6 = "You are a cautious ASR correction specialist. Start from the most likely hypothesis (Hypothesis 1), and only make small corrections when the audio clearly disagrees. Do not over-correct or introduce hallucinations. Use the other hypotheses only to resolve difficult sections."

def generate_all_sequences(model, prompt, generation_config):
    conversation = [{"from": "human", "value": prompt}]
    media, media_meta = extract_media(conversation, model.config)
    media_config = defaultdict(dict)
    for name in media:
        if name == "sound":
            sounds = process_sounds(media["sound"]).half()
            media[name] = [sound for sound in sounds]
            sound_feature_masks = process_sound_masks(media_meta["sound_feature_masks"]).half()
            media_meta["sound_feature_masks"] = [m for m in sound_feature_masks]
            sound_embed_masks = process_sound_masks(media_meta["sound_embed_masks"]).half()
            media_meta["sound_embed_masks"] = [m for m in sound_embed_masks]
        else:
            raise ValueError(f"Unsupported media type: {name}")

    input_ids = tokenize_conversation(conversation, model.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)
    output_ids = model.generate(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        media_meta=media_meta,
        generation_config=generation_config,
    )
    return [model.tokenizer.decode(out, skip_special_tokens=True).strip() for out in output_ids]

for i in tqdm(range(len(data))):
    if i < len(exist_data):
        continue
    example = data[i]

    sound = llava.Sound(example['audios'][0])
    conversations = example["conversations"]
    question = conversations[0]["value"]

    responses = generate_all_sequences(model, [sound, question], generation_config)
    for idx, resp in enumerate(responses, 1):
        print(f"{idx}) {resp}")
    # assert False
    if not no_audio:
        out_example = {
            "pred": responses,
            "gold": example['conversations'][1]['value'],
            "audio": example['audios'][0],
            "user_prompt": example['conversations'][0]['value']
        }
    else:
        out_example = {
            "pred": responses,
            "gold": example['conversations'][1]['value'],
            "user_prompt": example['conversations'][0]['value']
        }
    # print(out_example)
    # assert False
    exist_data.append(out_example)
    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(exist_data, f, indent=1, ensure_ascii=False)

# compute normalized WER, normalize to ignore punctuations
# pred_list = [item['pred'] for item in exist_data if item['gold'] != "ignore time segment in scoring"]
# gold_list = [item['gold'] for item in exist_data if item['gold'] != "ignore time segment in scoring"]
# wer_score = wer(gold_list, pred_list)
# print(f"WER: {wer_score}")
# wer_dict = {"wer": wer_score}

# with open(os.path.join(output_dir, "metric.json"), "w") as f:
#     json.dump(wer_dict, f, indent=1, ensure_ascii=False)
