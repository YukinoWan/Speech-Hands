import json
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer
from tqdm import tqdm
import ast
import re

dataset = ["ami", "voxpopuli", "tedlium", "spgispeech", "gigaspeech"]
# dataset = ["librispeech"]
type = ["train", "test"]
# type = ["train", "test_clean", "test_other"]

normalizer = BasicTextNormalizer()

def get_prompt(qwen_internal_output, whisper_hypos, audio):
    prompt = f"""
    You are an ASR correction model with access to three inputs:

    (1) The original audio;
    (2) Five transcription hypotheses from another ASR system (external);
    (3) Your own first-pass transcription (internal).

    Your task is to:
    - First decide whether your internal transcription is accurate enough.
    - If yes, output <no_ref> and the correct transcription.
    - If not, output <need_ref> and correct the transcription using the audio and the external hypotheses.

    Hypotheses:
    {whisper_hypos}

    Internal transcript:
    {qwen_internal_output}

    """
    return prompt.strip() + "\n\n"

def compute_wer(pred, gold):
    pred = normalize_text(pred)
    gold = normalize_text(gold)
    return jiwer.wer(pred, gold)

def normalize_text(text):
    cleaned = re.sub(r"<.*?>|\(.*?\)|\{.*?\}", "", text)
    return normalizer(cleaned)

for d in dataset:
    for t in type:
        with open(f"/mnt/home/zhenwan.nlp/zhen-openasr-n-best/parakeet_5_best_data/{d}_{t}_parakeet_5_best.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        with open(f"output/{d}_{t}_baseline_with_audio/output.json", "r") as f:
            baseline_data = json.load(f)
        print(len(data))
        print(len(baseline_data))
        # assert len(data) == len(baseline_data)
        baseline_correct = 0
        whisper_5_best_correct = 0
        for i in tqdm(range(len(data))):
            assert data[i]["audios"][0] == baseline_data[i]["audio"]
            # ori_whisper_5_best = "[" + data[i]["conversations"][0]["value"].split("[")[-1]
            # load string list to be list of strings
            # whisper_5_best = ast.literal_eval(ori_whisper_5_best)
            whisper_5_best = data[i]["whisper_5best"]
            baseline_pred = normalize_text(baseline_data[i]["pred"])
            gold = normalize_text(baseline_data[i]["gold"])
            for whisper in whisper_5_best:
                best_wer = 100
                wer = compute_wer(normalize_text(whisper), gold)
                if wer < best_wer:
                    best_wer = wer
            data[i]["conversations"][0]["value"] = get_prompt(baseline_pred, whisper_5_best, data[i]["audios"][0])
            # if compute_wer(baseline_pred, gold) == 0 or compute_wer(baseline_pred, gold) < best_wer:
            if compute_wer(baseline_pred, gold) < best_wer:
                baseline_correct += 1
                data[i]["conversations"][1]["value"] = "<no_ref>" + gold
            else:
                whisper_5_best_correct += 1
                data[i]["conversations"][1]["value"] = "<need_ref>" + gold
        print(f"Baseline correct: {baseline_correct}")
        print(f"Whisper 5 best correct: {whisper_5_best_correct}")
        with open(f"data/sharegpt_data/{d}_{t}_parakeet_5_best_with_audio.json", "w") as f:
            json.dump(data, f, indent=1)









