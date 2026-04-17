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

SPEECH_HANDS_DATA = os.environ.get("SPEECH_HANDS_DATA", "./data")

normalizer = BasicTextNormalizer()

task = sys.argv[1]
ckpt = sys.argv[2]
try:
    asr = sys.argv[3]
except:
    asr = None
# asr = sys.argv[3]
if "all" in ckpt:
    model_path = f"./target_dir/all-rag-with-audio-sharegpt-ckpt-{ckpt}"
elif "baseline" in ckpt:
    model_path = "Qwen/Qwen2.5-Omni-7B"
else:
    model_path = f"./target_dir/{task.split('_')[0]}-rag-with-audio-sharegpt-ckpt-{ckpt}"
#model_path = "Qwen/Qwen2.5-Omni-7B"
no_audio = False

print(no_audio)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2") 
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
from qwen_omni_utils import process_mm_info

if no_audio:
    with open(f"data/sharegpt_data_{ckpt}/{task}_{ckpt.split('-')[0]}_5_best_no_audio.json", "r") as f:
        data = json.load(f)
else:
    if "baseline" in ckpt:
        with open(f"{SPEECH_HANDS_DATA}/sharegpt/{asr}_prompt_v2/{task}_{asr}_5_best_with_audio.json", "r") as f:
            data = json.load(f)
    else:
        with open(f"data/sharegpt_data_{ckpt.split('-')[0]}_{ckpt.split('-')[1]}/{task}_{ckpt.split('-')[0]}_5_best_with_audio.json", "r") as f:
            data = json.load(f)

if "baseline" in ckpt:
    output_dir = f"{ckpt}_output/{task}_baseline_rag_with_audio"
else:
    output_dir = f"{ckpt}_output/{task}_with_audio_rag_ckpt-{ckpt}" if not no_audio else f"{ckpt}_output/{task}_no_audio_rag_ckpt-{ckpt}"
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



for i in tqdm(range(len(data))):
    # print(data[i]["conversations"][1]["value"])
    # assert False
    if i < len(exist_data):
        continue
    example = data[i]
    # print(normalizer(example['conversations'][1]['value']))
    # assert False
    # print(example['conversations'][0]['value'])
    if no_audio:
        conversation1 = [
            {'role': 'system', 'content': [
                {"type": "text", "text": system_prompt_v1}]},
            {"role": "user", "content": [
                {"type": "text", "text": example['conversations'][0]['value']},
            ]},
        ]
        conversations = [conversation1]
    else:
        conversation1 = [
            {'role': 'system', 'content': [
                {"type": "text", "text": system_prompt_v1}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": example['audios'][0]},
                {"type": "text", "text": example['conversations'][0]['value']},
            ]},
        ]
        conversations = conversation1
    

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)
    try:
        text_ids = model.generate(**inputs, use_audio_in_video=True, return_audio=False)
        text = processor.batch_decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except:
        text = [f"assistant\n{example['conversations'][1]['value']}"]

    print(text[0])
    # assert False
    if not no_audio:
        out_example = {
            "pred": text[0].split("assistant\n")[1].split("<|im_end|>")[0],
            "gold": example['conversations'][1]['value'],
            "audio": example['audios'][0],
            "system_prompt": system_prompt_v1,
            "user_prompt": example['conversations'][0]['value']
        }
    else:
        out_example = {
            "pred": text[0].split("assistant\n")[1].split("<|im_end|>")[0],
            "gold": example['conversations'][1]['value'],
            "system_prompt": system_prompt_v1,
            "user_prompt": example['conversations'][0]['value']
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
