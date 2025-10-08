import json
import jiwer

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sys
from tqdm import tqdm
import ast
from sklearn.metrics import precision_score, recall_score, f1_score
import re

normalizer = BasicTextNormalizer()

def compute_wer(pred, gold):
    pred = normalize_text(pred)
    gold = normalize_text(gold)
    return jiwer.wer(pred, gold)

def normalize_text(text):
    cleaned = re.sub(r"<.*?>|\(.*?\)|\{.*?\}", "", text)
    return normalizer(cleaned)

task = sys.argv[1]

pred_path = f"output/{task}_test_with_audio_rag_ckpt-full-v3/output.json"
input_path = f"data/sharegpt_data/{task}_test_whisper_5_best_with_audio.json"

with open(pred_path, "r") as f:
    preds = json.load(f)

with open(input_path, "r") as f:
    inputs = json.load(f)


labels_preds = []
labels_gold = []
for i in tqdm(range(len(preds))):
    pred_example = preds[i]
    input_example = inputs[i]
    gold = pred_example["gold"]
    pred = pred_example["pred"]
    prompt = input_example["conversations"][0]["value"]
    baseline = prompt.split("Internal transcript:")[1].strip()
    ori_whisper_5_best = "[" + prompt.split("Internal transcript:")[0].strip().split("[")[-1]
    # load string list to be list of strings
    whisper_5_best = ast.literal_eval(ori_whisper_5_best)
    best_wer = 100
    for whisper in whisper_5_best:
        wer = compute_wer(whisper, gold)
        if wer < best_wer:
            best_wer = wer
    if compute_wer(pred, baseline) == 0:
        labels_preds.append(0)
        if compute_wer(pred, gold) == 0:
            labels_gold.append(0)
        elif compute_wer(pred, gold) < best_wer or compute_wer(pred, gold) == best_wer:
            labels_gold.append(0)
        else:
            labels_gold.append(1)
    else:
        labels_preds.append(1)
        if compute_wer(pred, gold) == 0:
            labels_gold.append(1)
        elif compute_wer(pred, gold) < compute_wer(baseline, gold) or compute_wer(pred, gold) == compute_wer(baseline, gold) or best_wer < compute_wer(baseline, gold) or best_wer == compute_wer(baseline, gold):
            labels_gold.append(1)
        else:
            labels_gold.append(0)
    # based on labels_preds and labels_gold, calculate the precision, recall and f1 score
precision = precision_score(labels_preds, labels_gold)
recall = recall_score(labels_preds, labels_gold)
f1 = f1_score(labels_preds, labels_gold)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")

need_ref_count = 0
no_ref_count = 0

train_path = f"data/sharegpt_data/{task}_train_whisper_5_best_with_audio.json"
with open(train_path, "r") as f:    
    train_data = json.load(f)

for i in tqdm(range(len(train_data))):
    response = train_data[i]["conversations"][1]["value"]
    if "<need_ref>" in response:
        need_ref_count += 1
    elif "<no_ref>" in response:
        no_ref_count += 1
print(f"Need ref count: {need_ref_count}")
print(f"No ref count: {no_ref_count}")
print(f"Total count: {need_ref_count + no_ref_count}")
