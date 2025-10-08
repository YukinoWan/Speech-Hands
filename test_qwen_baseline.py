import soundfile as sf
from io import BytesIO
from urllib.request import urlopen
from qwen_omni_utils import process_vision_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
import json
import os
from tqdm import tqdm
from jiwer import wer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sys

normalizer = BasicTextNormalizer()

task = sys.argv[1]
#model_path = f"./target_dir/{task}-whisper-5-best-with-audio-sharegpt-ckpt-{ckpt}"
model_path = "Qwen/Qwen2.5-Omni-7B"
no_audio = False if sys.argv[2] == "False" else True
no_audio = False
# print(no_audio)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto") 
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
from qwen_omni_utils import process_mm_info

if no_audio:
    with open(f"data/sharegpt_data_v6/{task}_test_whisper_5_best_no_audio.json", "r") as f:
        data = json.load(f)
else:
    with open(f"data/sharegpt_data_v6/{task}_test_whisper_5_best_with_audio.json", "r") as f:
        data = json.load(f)

#output_dir = f"output/{task}_test_with_audio_whisper_5_best" if not no_audio else f"output/{task}_test_no_audio_whisper_5_best"
output_dir = f"output/{task}_full_test_baseline_with_audio"
os.makedirs(output_dir, exist_ok=True)

try:
    with open(os.path.join(output_dir, "output.json"), "r") as f:
        exist_data = json.load(f)
except:
    exist_data = []


for i in tqdm(range(len(data))):
    if i < len(exist_data):
        continue
    example = data[i]
    if no_audio:
        conversation1 = [
            {'role': 'system', 'content': [
                {"type": "text", "text": 'You are a speech recognition model.'}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe the English audio into text without any punctuation marks."},
            ]},
        ]
        conversations = [conversation1]
    else:
        conversation1 = [
            {'role': 'system', 'content': [
                {"type": "text", "text": 'You are a speech recognition model.'}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": example['audios'][0]},
                {"type": "text", "text": "Transcribe the English audio into text without any punctuation marks."},
            ]},
        ]
        conversations = conversation1
    

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)
    try:
        text_ids = model.generate(**inputs, use_audio_in_video=True, return_audio=False)
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except:
        text = [f"assistant\n{example['conversations'][1]['value']}"]
    print(text)
    if not no_audio:
        out_example = {
            "pred": normalizer(text[0].split("assistant\n")[1]),
            "gold": normalizer(example['conversations'][1]['value']),
            "audio": example['audios'][0]
        }
    else:
        out_example = {
            "pred": normalizer(text[0].split("assistant\n")[1]),
            "gold": normalizer(example['conversations'][1]['value']),
        }
    exist_data.append(out_example)
    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(exist_data, f, indent=1, ensure_ascii=False)

# compute normalized WER, normalize to ignore punctuations
pred_list = [item['pred'] for item in exist_data if item['gold'] != "ignore time segment in scoring"]
gold_list = [item['gold'] for item in exist_data if item['gold'] != "ignore time segment in scoring"]
wer_score = wer(gold_list, pred_list)
print(f"WER: {wer_score}")
wer_dict = {"wer": wer_score}

with open(os.path.join(output_dir, "metric.json"), "w") as f:
    json.dump(wer_dict, f, indent=1, ensure_ascii=False)
