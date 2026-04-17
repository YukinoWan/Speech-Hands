import json
import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sys
import re

from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

SPEECH_HANDS_DATA = os.environ.get("SPEECH_HANDS_DATA", "./data")

normalizer = BasicTextNormalizer()

def normalize_text(text):
    cleaned = re.sub(r"<.*?>|\(.*?\)|\{.*?\}", "", text)
    return cleaned

task = sys.argv[1]
ckpt = sys.argv[2]

asr_model = ckpt.split("-")[0]

with open(f"parakeet-ger_rag_output/{task}_with_audio_rag_ckpt-{ckpt}/output.json", "r") as f:
    data = json.load(f)
with open(f"{SPEECH_HANDS_DATA}/asr_nbest/{asr_model}/{task}_{asr_model}_5_best.jsonl", "r") as f:
    nbest_data = [json.loads(line) for line in f]
with open(f"{SPEECH_HANDS_DATA}/asr_nbest/{asr_model}/{task}_{asr_model}_5_best.jsonl", "r") as f:
    w1_best_data = [json.loads(line) for line in f]
with open(f"data/sharegpt_data_v6/{task}_whisper_5_best_with_audio.json", "r") as f:
    gold_data = json.load(f)
with open(f"output/{task}_baseline_with_audio/output.json", "r") as f:
    baseline_data = json.load(f)
with open(f"parakeet_output/{task}_baseline_rag_with_audio/output.json", "r") as f:
    ger_data = json.load(f)


# w1_best_data = [x for x in w1_best_data if "ignore time segment in scoring" not in x["messages"][1]["content"]]
# baseline_data = [x for x in baseline_data if "ignore time segment in scoring" != x["gold"]]
# ger_data = [x for x in ger_data if "ignore time segment in scoring" != x["gold"]]
final_list = []
gold_list = []
w1_best_list = []
baseline_list = []
ger_list = []

pred_token_list = []
gold_token_list = []
# assert len(w1_best_data) == len(baseline_data) == len(ger_data) == len(data)
for i in range(len(data)):
    whisper_1_best = normalize_text(w1_best_data[i]["whisper_5best"])
    # whisper_5_best = [whisper_1_best] + nbest_data[i]["whisper_5best"]
    baseline_pred = normalize_text(baseline_data[i]["pred"])
    ger_pred = normalize_text(ger_data[i]["pred"])
    # assert False
    ori_gold = gold_data[i]["conversations"][1]["value"]
    if "ignore_time_segment_in_scoring" in ori_gold:
        continue
    gold = normalize_text(gold_data[i]["conversations"][1]["value"])
    
    response = data[i]["pred"]

    pred_token = response.split(">")[0] + ">"
    gold_token = data[i]["gold"].split(">")[0] + ">"
    try:
        pred = normalizer(response.split(">")[1])
    except:
        pred = gold
        pred_token = gold_token
        # assert False


    

    # gold = normalizer(data[i]["gold"].split(">")[1])


    # w1_best = normalizer(w1_best_data[i]["whisper_5best"])
    # nbest = nbest_data[i]["whisper_5best"]
    # baseline = normalizer(baseline_data[i]["pred"])
    # ger = normalizer(ger_data[i]["pred"])

    # if pred_token == "<internal>" and gold_token == "<internal>" and w1_best != gold and ger != gold:
    #     print("pred_token:", pred_token)
    #     print("pred:", pred)
    #     print("ger:", ger)
    #     print("gold:", gold)
    #     print("w1_best:", w1_best)
    #     print("baseline:", baseline)
    #     print("--------------------------------")
    # if pred_token == "<external>" and gold_token == "<external>":
    #     print("pred_token:", pred_token)
    #     print("pred:", pred)
    #     print("ger:", ger)
    #     print("gold:", gold)
    #     print("w1_best:", w1_best)
    #     print("baseline:", baseline)
    #     print("--------------------------------")

    # if pred_token == "<rewrite>" and gold_token == "<rewrite>" and pred != gold and w1_best != gold:
    #     print("pred_token:", pred_token)
    #     print("pred:", pred)
    #     print("ger:", ger)
    #     print("gold:", gold)
    #     print("w1_best:", w1_best)
    #     print("baseline:", baseline)
    #     print("--------------------------------")

    if pred_token == "<internal>":
        final_pred = baseline_pred
    elif pred_token == "<external>":
        final_pred = pred
    elif pred_token == "<rewrite>":
        final_pred = ger_pred
    else:
        final_pred = pred


    w1_best_list.append(normalizer(whisper_1_best))
    baseline_list.append(normalizer(baseline_pred))
    ger_list.append(normalizer(ger_pred))
    final_list.append(normalizer(final_pred))
    pred_token_list.append(pred_token.strip())
    gold_token_list.append(gold_token.strip())
    gold_list.append(normalizer(gold))



print("jiwer.wer(final_list, gold_list):", jiwer.wer(final_list, gold_list))
print("jiwer.wer(w1_best_list, gold_list):", jiwer.wer(w1_best_list, gold_list))
print("jiwer.wer(baseline_list, gold_list):", jiwer.wer(baseline_list, gold_list))
print("jiwer.wer(ger_list, gold_list):", jiwer.wer(ger_list, gold_list))

print("classification_report:")
print(classification_report(gold_token_list, pred_token_list))
print("confusion_matrix:")
print(confusion_matrix(gold_token_list, pred_token_list))
