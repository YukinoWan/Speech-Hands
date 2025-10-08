#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WER with mandatory Whisper BasicTextNormalizer and list worst examples."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON (array) or JSONL with fields 'pred' and 'gold'.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write outputs (default: <input_dir>/wer_analysis).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="How many worst-WER examples to save (default: 100).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, also save examples with WER >= threshold.",
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
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def get_fields(obj: Dict[str, Any]) -> Tuple[str, str, str]:
    hyp = obj.get("pred") or obj.get("hyp") or obj.get("prediction") or obj.get("text") or ""
    ref = obj.get("gold") or obj.get("ref") or obj.get("reference") or obj.get("target") or ""
    audio = obj.get("audio") or obj.get("audio_path") or obj.get("path") or ""
    return str(hyp), str(ref), str(audio)


def write_tsv(path: str, rows: List[Tuple[int, float, int, int, str, str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as w:
        w.write("index\twer\tref_len\thyp_len\taudio\tgold\tpred\n")
        for idx, wer, ref_len, hyp_len, audio, gold, pred in rows:
            # sanitize tabs/newlines
            audio_s = audio.replace("\t", " ").replace("\n", " ")
            gold_s = gold.replace("\t", " ").replace("\n", " ")
            pred_s = pred.replace("\t", " ").replace("\n", " ")
            w.write(f"{idx}\t{wer:.6f}\t{ref_len}\t{hyp_len}\t{audio_s}\t{gold_s}\t{pred_s}\n")


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    outdir = os.path.abspath(args.outdir) if args.outdir else os.path.join(os.path.dirname(input_path), "wer_analysis")
    os.makedirs(outdir, exist_ok=True)

    records = read_records(input_path)
    if not records:
        print("No records found.")
        print(f"Input: {input_path}")
        return

    normalizer = BasicTextNormalizer()

    per_sample: List[Tuple[int, float, int, int, str, str, str]] = []
    total_dist = 0
    total_ref = 0
    analyzed = 0
    skipped = 0

    for idx, obj in enumerate(records):
        hyp_raw, ref_raw, audio = get_fields(obj)
        hyp_norm = normalize(hyp_raw, normalizer)
        ref_norm = normalize(ref_raw, normalizer)
        hyp_toks = tokenize(hyp_norm)
        ref_toks = tokenize(ref_norm)
        n = len(ref_toks)
        m = len(hyp_toks)
        if n == 0:
            # skip from average; still record with conventional WER
            wer = 0.0 if m == 0 else 1.0
            per_sample.append((idx, wer, n, m, audio, ref_raw, hyp_raw))
            skipped += 1
            continue
        dist = levenshtein_distance(ref_toks, hyp_toks)
        wer = dist / float(n)
        per_sample.append((idx, wer, n, m, audio, ref_raw, hyp_raw))
        total_dist += dist
        total_ref += n
        analyzed += 1

    avg_wer = (total_dist / float(total_ref)) if total_ref > 0 else 0.0

    per_path = os.path.join(outdir, "per_sample.tsv")
    write_tsv(per_path, per_sample)

    sorted_rows = sorted(per_sample, key=lambda r: (r[1], r[2]), reverse=True)
    topk = max(1, args.topk)
    top_rows = sorted_rows[:topk]
    top_path = os.path.join(outdir, f"top_worst_{topk}.tsv")
    write_tsv(top_path, top_rows)

    threshold_path = None
    if args.threshold is not None:
        thr_rows = [r for r in sorted_rows if r[1] >= args.threshold]
        threshold_path = os.path.join(outdir, f"wer_ge_{args.threshold:.2f}.tsv")
        write_tsv(threshold_path, thr_rows)

    print("WER analysis completed.")
    print(f"Input file: {input_path}")
    print(f"Output dir: {outdir}")
    print(f"Analyzed: {analyzed}, Skipped (empty ref): {skipped}")
    print(f"Average WER: {avg_wer:.6f} ({avg_wer * 100:.2f}%)")
    print(f"Per-sample TSV: {per_path}")
    print(f"Top-{topk} worst TSV: {top_path}")
    if threshold_path:
        print(f"Threshold TSV (>= {args.threshold:.2f}): {threshold_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130) 