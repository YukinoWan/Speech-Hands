import json
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer
from tqdm import tqdm
import ast
import re
import numpy as np

dataset = ["ami", "voxpopuli", "tedlium", "spgispeech", "gigaspeech"]
# dataset = ["tedlium","gigaspeech"]
# dataset = ["librispeech"]
type = ["train", "test"]
# type = ["train", "test_clean", "test_other"]

normalizer = BasicTextNormalizer()

def get_prompt(qwen_internal_output, whisper_hypos, audio):
    # prompt = f"""
    # You are an ASR correction model with access to three inputs:

    # (1) The original audio;
    # (2) Five transcription hypotheses from another ASR system (external);
    # (3) Your own first-pass transcription (internal).

    # Your task is to:
    # - First decide whether your internal transcription is accurate enough.
    # - If yes, output <no_ref> and the correct transcription.
    # - If not, output <need_ref> and correct the transcription using the audio and the external hypotheses.

    # Hypotheses:
    # {whisper_hypos}

    # Internal transcript:
    # {qwen_internal_output}

    # """

#     prompt = f"""You are an ASR correction model with access to:

# (1) The original audio;
# (2) Transcription hypotheses from another ASR system (external);
# (3) Your own first-pass transcription (internal).

# Your task is to:
# - Compare the quality of your internal transcript with the external options.
# - If your internal output is the most accurate, output <internal> followed by the transcription.
# - If one of the external hypotheses is clearly better, output <external> followed by that transcription.
# - If both are inaccurate but you can revise a better answer by considering audio + external hypotheses, output <rewrite> and the revised transcription.

# Hypotheses:
# {whisper_hypos}

# Internal transcript:
# {qwen_internal_output}

# """

    prompt = f"""You are an ASR correction model with access to:

(1) The original audio;
(2) The transcription hypotheses from another ASR system (external);
(3) Your own first-pass transcription (internal).

Your task is to:
- Compare the quality of your internal transcript with the external options.
- If your internal output is the most accurate, output <internal> followed by the transcription.
- If one of the external hypotheses is clearly better, output <external> followed by that transcription.
- If both are inaccurate but you can revise a better answer by considering audio + external hypotheses, output <rewrite> and the revised transcription.

Hypotheses:
{whisper_hypos}

Internal transcript:
{qwen_internal_output}

"""
    return prompt.strip() + "\n\n"

def compute_wer(pred, gold):
    pred = normalizer(pred)
    gold = normalizer(gold)
    return jiwer.wer(pred, gold)

def normalize_text(text):
    cleaned = re.sub(r"<.*?>|\(.*?\)|\{.*?\}", "", text)
    return cleaned

for d in dataset:
    for t in type:
        output_data = []
        with open(f"/mnt/home/zhenwan.nlp/zhen-openasr-n-best/parakeet_5_best_data/{d}_{t}_parakeet_5_best.jsonl", "r") as f:
            nbest_data = [json.loads(line) for line in f]
        with open(f"/mnt/home/zhenwan.nlp/zhen-openasr-n-best/parakeet_data/{d}_{t}_parakeet_5_best.jsonl", "r") as f:
            w1_best_data = [json.loads(line) for line in f]
        with open(f"data/sharegpt_data_v6/{d}_{t}_whisper_5_best_with_audio.json", "r") as f:
            gold_data = json.load(f)
        with open(f"output/{d}_{t}_baseline_with_audio/output.json", "r") as f:
            baseline_data = json.load(f)
        with open(f"output/{d}_{t}_ger_with_audio_ckpt-parakeet_v2_ger/output.json", "r") as f:
            ger_data = json.load(f)
        print(len(nbest_data))
        print(len(baseline_data))
        print(len(ger_data))
        # assert len(data) == len(baseline_data)
        baseline_correct = 0
        whisper_1_best_correct = 0
        ger_correct = 0

        baseline_list = []
        whisper_1_best_list = []
        whisper_5_best_list = []
        ger_list = []
        gold_list = []
        best_wer_list = []
        out_wer_list = []
        for i in tqdm(range(len(nbest_data))):
            assert nbest_data[i]["audios"][0] == baseline_data[i]["audio"]
            # ori_whisper_5_best = "[" + data[i]["conversations"][0]["value"].split("[")[-1]
            # load string list to be list of strings
            # whisper_5_best = ast.literal_eval(ori_whisper_5_best)
            
            whisper_1_best = normalize_text(w1_best_data[i]["whisper_5best"])
            whisper_5_best = [whisper_1_best] + nbest_data[i]["whisper_5best"]
            baseline_pred = normalize_text(baseline_data[i]["pred"])
            ger_pred = normalize_text(ger_data[i]["pred"])
            # assert False
            gold = normalize_text(gold_data[i]["conversations"][1]["value"])
            if gold == "ignore_time_segment_in_scoring":
                # print(gold)
                continue
            # print(gold)
            
            best_wer = 100
            for whisper in whisper_5_best:
                
                wer = compute_wer(normalize_text(whisper), gold)
                if wer < best_wer:
                    best_wer = wer
                    best_whisper = whisper
            assert best_wer <= compute_wer(whisper_1_best, gold)
            whisper_5_best_list.append(normalizer(best_whisper))
            tmp_data = {}
            tmp_data["audios"] = nbest_data[i]["audios"]
            tmp_data["conversations"] = [{"from": "human", "value": get_prompt(baseline_pred, whisper_5_best, nbest_data[i]["audios"][0])}, {"from": "gpt", "value": ""}]
            
            # nbest_data[i]["conversations"][0]["value"] = get_prompt(baseline_pred, whisper_5_best, nbest_data[i]["audios"][0])
            # if compute_wer(baseline_pred, gold) == 0 or compute_wer(baseline_pred, gold) < best_wer:
            if compute_wer(baseline_pred, gold) == 0 or (compute_wer(baseline_pred, gold) <= compute_wer(best_whisper, gold) and compute_wer(baseline_pred, gold) <= compute_wer(ger_pred, gold)):
                baseline_correct += 1
                best_wer_list.append(normalizer(baseline_pred))
                out_wer_list.append(compute_wer(baseline_pred, gold))
                tmp_data["conversations"][1]["value"] = "<internal>" + gold
            elif compute_wer(best_whisper, gold) <= compute_wer(baseline_pred, gold) and compute_wer(best_whisper, gold) <= compute_wer(ger_pred, gold):
                whisper_1_best_correct += 1
                best_wer_list.append(normalizer(best_whisper))
                out_wer_list.append(compute_wer(best_whisper, gold))
                tmp_data["conversations"][1]["value"] = "<external>" + gold
            else:
                ger_correct += 1
                best_wer_list.append(normalizer(ger_pred))
                out_wer_list.append(compute_wer(ger_pred, gold))
                tmp_data["conversations"][1]["value"] = "<rewrite>" + gold
            gold_list.append(normalizer(gold))
            baseline_list.append(normalizer(baseline_pred))
            whisper_1_best_list.append(normalizer(whisper_1_best))
            ger_list.append(normalizer(ger_pred))
            # whisper_5_best_list.append(best_wer)
            output_data.append(tmp_data)

        
        print(f"Baseline correct: {baseline_correct}")
        print(f"Whisper 1 best correct: {whisper_1_best_correct}")
        print(f"Ger correct: {ger_correct}")

        print(f"Baseline best wer: {jiwer.wer(baseline_list, gold_list)}")
        print(f"Whisper 1 best best wer: {jiwer.wer(whisper_1_best_list, gold_list)}")
        print(f"Ger wer: {jiwer.wer(ger_list, gold_list)}")
        print(f"Best wer: {jiwer.wer(best_wer_list, gold_list)}")
        print(f"Whisper 5 best wer: {jiwer.wer(whisper_5_best_list, gold_list)}")
        print(f"Out wer: {sum(out_wer_list) / len(out_wer_list)}")
        with open(f"data/sharegpt_data_parakeet_ger_rag/{d}_{t}_parakeet_5_best_with_audio.json", "w") as f:
            json.dump(output_data, f, indent=1)









