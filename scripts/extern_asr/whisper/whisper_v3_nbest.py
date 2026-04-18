"""Generate Whisper-large-v3 N-best hypotheses for a Speech-Hands ASR split.

Reads a ShareGPT-format JSONL where each line has at minimum an `id` and a
`sound` field pointing to an audio file. Writes one JSON per split with the
5-best transcription hypotheses produced by beam search (num_beams=20,
num_return_sequences=5).

Typical invocation:

    SPEECH_HANDS_DATA=/path/to/data \
    python scripts/extern_asr/whisper/whisper_v3_nbest.py <input.jsonl> <output.json>

The output JSON is consumed by `scripts/process_asr_nbest.py` after being
converted to the repo's asr_nbest/ naming scheme.
"""

import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


SPEECH_HANDS_DATA = os.environ.get("SPEECH_HANDS_DATA", "./data")

sys.setrecursionlimit(3000)


def main(input_path: str, output_path: str, model_id: str = "openai/whisper-large-v3"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    generate_kwargs = {
        "num_beams": 20,
        "num_return_sequences": 5,
    }

    with open(input_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    examples = []
    for item in tqdm(dataset):
        sound = item["sound"]
        result = pipe(sound, generate_kwargs=generate_kwargs)
        example = {
            "id": item["id"],
            "audios": [sound],
            "whisper_5best": [r["text"] for r in result] if isinstance(result, list) else [result["text"]],
        }
        if "conversations" in item:
            example["messages"] = [
                {"role": "user", "content": item["conversations"][0]["value"]},
                {"role": "assistant", "content": item["conversations"][1]["value"]},
            ]
        examples.append(example)

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=1)
    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
