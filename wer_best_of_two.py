#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick per-sample lower WER between two model outputs (Whisper BasicTextNormalizer)."
    )
    parser.add_argument("--input1", required=True, help="First model output JSON/JSONL (fields: pred, gold, audio)")
    parser.add_argument("--input2", required=True, help="Second model output JSON/JSONL (same ordering as input1)")
    parser.add_argument(
        "--ref",
        default=None,
        help="Optional reference JSON/JSONL to include the original input (ref) in output.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write outputs (default: <dir_of_input1>/best_of_two)",
    )
    return parser.parse_args()


def read_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return []
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            return []
        # JSONL
        records: List[Dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
        return records


def get_fields(obj: Dict[str, Any]) -> Tuple[str, str, str]:
    hyp = obj.get("pred") or obj.get("hyp") or obj.get("prediction") or obj.get("text") or ""
    ref = obj.get("gold") or obj.get("ref") or obj.get("reference") or obj.get("target") or ""
    audio = obj.get("audio") or obj.get("audio_path") or obj.get("path") or ""
    return str(hyp), str(ref), str(audio)


def extract_ref_input_and_audio(obj: Dict[str, Any]) -> Tuple[str, str]:
    audio = obj.get("audio") or obj.get("audio_path") or obj.get("path") or ""
    input_text = ""
    conv = obj.get("conversations")
    if isinstance(conv, list) and conv:
        for item in conv:
            if isinstance(item, dict) and item.get("from") == "human":
                val = item.get("value")
                # mirror user's adjustment: keep bracketed suffix
                val = "[" + val.split("[")[-1]
                if isinstance(val, str):
                    input_text = val
                    break
    if not input_text:
        input_text = (
            obj.get("input") or obj.get("prompt") or obj.get("instruction") or obj.get("system") or ""
        )
    return str(input_text), str(audio)


def load_ref_maps(ref_path: str) -> Tuple[List[str], Dict[str, str]]:
    try:
        records = read_records(ref_path)
    except Exception:
        return [], {}
    index_to_input: List[str] = []
    audio_to_input: Dict[str, str] = {}
    for obj in records:
        input_text, audio = extract_ref_input_and_audio(obj)
        index_to_input.append(input_text)
        if audio:
            audio_to_input[audio] = input_text
    return index_to_input, audio_to_input


def normalize(text: Optional[str], normalizer: BasicTextNormalizer) -> str:
    if not text:
        return ""
    return normalizer(text)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def write_tsv(path: str, rows: List[Tuple[int, float, int, int, str, str, str, str, str, str]]):
    with open(path, "w", encoding="utf-8") as w:
        w.write("index\twer\tdist\tref_len\thyp_len\tbest\taudio\tref\tgold\tpred1\tpred2\n")
        for idx, wer, dist, ref_len, hyp_len, best, audio, ref_text, gold, pred1, pred2 in rows:
            def san(x: str) -> str:
                return x.replace("\t", " ").replace("\n", " ")
            w.write(
                f"{idx}\t{wer:.6f}\t{dist}\t{ref_len}\t{hyp_len}\t{best}\t{san(audio)}\t{san(ref_text)}\t{san(gold)}\t{san(pred1)}\t{san(pred2)}\n"
            )


def main() -> None:
    args = parse_args()
    input1 = os.path.abspath(args.input1)
    input2 = os.path.abspath(args.input2)
    outdir = os.path.abspath(args.outdir) if args.outdir else os.path.join(os.path.dirname(input1), "best_of_two")
    os.makedirs(outdir, exist_ok=True)

    recs1 = read_records(input1)
    recs2 = read_records(input2)

    n = min(len(recs1), len(recs2))
    if n == 0:
        print("No overlapping records to compare.")
        print(f"Input1: {input1}")
        print(f"Input2: {input2}")
        return
    if len(recs1) != len(recs2):
        print(f"Warning: different lengths (input1={len(recs1)}, input2={len(recs2)}); truncating to {n}.")

    # load optional reference map
    index_to_input: List[str] = []
    audio_to_input: Dict[str, str] = {}
    if args.ref:
        ref_path = os.path.abspath(args.ref)
        index_to_input, audio_to_input = load_ref_maps(ref_path)

    normalizer = BasicTextNormalizer()

    rows: List[Tuple[int, float, int, int, str, str, str, str, str, str]] = []
    total_dist = 0
    total_ref = 0

    for idx in range(n):
        hyp1_raw, ref1_raw, audio1 = get_fields(recs1[idx])
        hyp2_raw, ref2_raw, audio2 = get_fields(recs2[idx])
        gold_raw = ref1_raw if ref1_raw else ref2_raw
        audio = audio1 or audio2

        # align ref text
        ref_text = ""
        if audio and audio in audio_to_input:
            ref_text = audio_to_input[audio]
        elif idx < len(index_to_input):
            ref_text = index_to_input[idx]

        gold_norm = normalize(gold_raw, normalizer)
        hyp1_norm = normalize(hyp1_raw, normalizer)
        hyp2_norm = normalize(hyp2_raw, normalizer)

        ref_toks = tokenize(gold_norm)
        hyp1_toks = tokenize(hyp1_norm)
        hyp2_toks = tokenize(hyp2_norm)

        ref_len = len(ref_toks)
        if ref_len == 0:
            dist1 = 0 if len(hyp1_toks) == 0 else len(hyp1_toks)
            dist2 = 0 if len(hyp2_toks) == 0 else len(hyp2_toks)
            wer1 = 0.0 if len(hyp1_toks) == 0 else 1.0
            wer2 = 0.0 if len(hyp2_toks) == 0 else 1.0
        else:
            dist1 = levenshtein_distance(ref_toks, hyp1_toks)
            dist2 = levenshtein_distance(ref_toks, hyp2_toks)
            wer1 = dist1 / float(ref_len)
            wer2 = dist2 / float(ref_len)

        if (wer1, dist1) <= (wer2, dist2):
            best = "1"
            wer = wer1
            dist = dist1
            hyp_len = len(hyp1_toks)
        else:
            best = "2"
            wer = wer2
            dist = dist2
            hyp_len = len(hyp2_toks)

        rows.append((idx, wer, dist, ref_len, hyp_len, best, audio, ref_text, gold_raw, hyp1_raw, hyp2_raw))

        if ref_len > 0:
            total_dist += dist
            total_ref += ref_len

    avg_wer = (total_dist / float(total_ref)) if total_ref > 0 else 0.0

    out_tsv = os.path.join(outdir, "best_of_two.tsv")
    write_tsv(out_tsv, rows)

    print("Best-of-two WER analysis completed.")
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    if args.ref:
        print(f"Reference: {os.path.abspath(args.ref)}")
    print(f"Output: {out_tsv}")
    print(f"Average WER (best-of-two): {avg_wer:.6f} ({avg_wer * 100:.2f}%)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130) 