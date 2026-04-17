import json
import os
import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sys

SPEECH_HANDS_DATA = os.environ.get("SPEECH_HANDS_DATA", "./data")

normalizer = BasicTextNormalizer()

task = sys.argv[1]

data_path = f"output/{task}_with_audio_rag_ckpt-full-v3/output.json"

qwen_data_path = f"output/{task}_baseline_with_audio/output.json"

with open(data_path, "r") as f:
    data = json.load(f)

with open(qwen_data_path, "r") as f:
    qwen_data = json.load(f)

canary_data_path = f"{SPEECH_HANDS_DATA}/asr_nbest/parakeet/{task}_parakeet_5_best.jsonl"

with open(canary_data_path, "r") as f:
    canary_data = [json.loads(line) for line in f]

parakeet_data_path = f"{SPEECH_HANDS_DATA}/asr_nbest/parakeet/{task}_parakeet_5_best.jsonl"

with open(parakeet_data_path, "r") as f:
    parakeet_data = [json.loads(line) for line in f]

print("len(canary_data):", len(canary_data))
print("len(data):", len(data))
print("len(qwen_data):", len(qwen_data))
ref_list = []
hyp_list = []
qwen_list = []
canary_list = []
parakeet_list = []
data = data[:20000]
for i in range(len(data)):
    item = data[i]
    if "ignore time segment in scoring" in item["gold"]:
        continue
    reference = normalizer(item["gold"]).strip()
    hypothesis = normalizer(item["pred"]).strip()
    qwen = normalizer(qwen_data[i]["pred"]).strip()
    if jiwer.wer(hypothesis, qwen) > 1.0:
        hypothesis = qwen
    ref_list.append(reference)
    hyp_list.append(hypothesis)
    qwen_list.append(qwen)
    canary_list.append(normalizer(canary_data[i]["whisper_5best"]))
    parakeet_list.append(normalizer(parakeet_data[i]["whisper_5best"]))
    print(hypothesis)
    print(reference)
    print(qwen)
    print(normalizer(canary_data[i]["whisper_5best"]))
    print(normalizer(parakeet_data[i]["whisper_5best"]))
    print("--------------------------------")
    # assert False

print("hypothesis vs reference:", jiwer.wer(hyp_list, ref_list))
print("qwen vs reference:", jiwer.wer(qwen_list, ref_list))
print("canary vs reference:", jiwer.wer(canary_list, ref_list))
print("parakeet vs reference:", jiwer.wer(parakeet_list, ref_list))

