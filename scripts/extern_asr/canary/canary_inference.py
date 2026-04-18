"""Generate Canary-1b N-best hypotheses via NVIDIA NeMo.

NeMo's canary-1b does not expose a first-class beam-search API, so we
approximate an N-best pool by sampling the model multiple times with
temperature > 0. The default is 5 samples with temperature 0.3, which
matches the paper's Canary setup.

Requires:
    pip install nemo_toolkit['asr']

Typical invocation:

    SPEECH_HANDS_DATA=/path/to/data \
    python scripts/extern_asr/canary/canary_inference.py <split>

where <split> is one of: ami / tedlium / voxpopuli / earnings22 / medasr /
gigaspeech / spgispeech / librispeech (use the key your local loader
understands).

The output is written to `$SPEECH_HANDS_DATA/asr_nbest/canary/<split>_canary_5_best.json`
and then converted to the training jsonl format by
`scripts/process_asr_1best.py` / `scripts/process_asr_nbest.py`.
"""

import json
import os
import sys

from datasets import load_dataset
from nemo.collections.asr.models import EncDecMultiTaskModel
from tqdm import tqdm


SPEECH_HANDS_DATA = os.environ.get("SPEECH_HANDS_DATA", "./data")


def load_split(name: str):
    """Loaders for the subsets we ran Canary on.

    Extend this for additional OpenASR benchmarks; each loader just needs
    to return an iterable of dicts with a local audio path (`audio_path`)
    and a reference transcript (`text`).
    """
    if name == "voxpopuli":
        ds = load_dataset("facebook/voxpopuli", "en", split="test")
        return [{"audio_path": x["audio"]["path"], "text": x["raw_text"]} for x in ds]
    if name == "tedlium":
        ds = load_dataset("LIUM/tedlium", "release1", split="test")
        return [{"audio_path": x["audio"]["path"], "text": x["text"]} for x in ds]
    if name == "earnings22":
        ds = load_dataset("anton-l/earnings22_robust", "all", split="test")
        return [{"audio_path": x["audio"]["path"], "text": x["sentence"]} for x in ds]
    if name == "medasr":
        ds = load_dataset("jarvisx17/Medical-ASR-EN", split="train")
        return [{"audio_path": x["audio"]["path"], "text": x["transcription"]} for x in ds]
    raise ValueError(f"Unknown split: {name}")


def get_canary_nbest(audio_paths, num_samples: int = 5, temperature: float = 0.3):
    model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
    decode_cfg = model.cfg.decoding
    decode_cfg.temperature = temperature
    model.change_decoding_strategy(decode_cfg)

    outputs = []
    for path in tqdm(audio_paths):
        samples = []
        for _ in range(num_samples):
            result = model.transcribe(audio=path)
            samples.append(result[0])
        outputs.append(samples)
    return outputs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    split = sys.argv[1]

    data = load_split(split)
    audio_paths = [x["audio_path"] for x in data]
    nbest = get_canary_nbest(audio_paths)

    out = []
    for i, sample in enumerate(data):
        out.append({
            "id": i,
            "audio": sample["audio_path"],
            "gold": sample["text"],
            "canary_nbest": nbest[i],
        })

    out_dir = os.path.join(SPEECH_HANDS_DATA, "asr_nbest", "canary")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_canary_5_best.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {len(out)} examples to {out_path}")


if __name__ == "__main__":
    main()
