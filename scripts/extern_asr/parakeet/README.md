# Parakeet-TDT N-best generation

The paper's main external ASR is `nvidia/parakeet-tdt-0.6b-v3`. Inference
is done entirely through [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)'s
shipped `transcribe_speech.py` script — there is no custom code in this
repo, because NeMo already handles beam search, N-best return, and
manifest I/O.

## Install NeMo

```bash
pip install nemo_toolkit['asr']
# For best performance install a matching CUDA-enabled torch first;
# see https://github.com/NVIDIA/NeMo#installation.
```

Clone NeMo for the CLI script:

```bash
git clone https://github.com/NVIDIA/NeMo.git ~/NeMo
```

## Build a NeMo manifest

Each line is one utterance:

```json
{"audio_filepath": "/path/to/audio.wav", "duration": 12.3, "text": "reference transcript"}
```

For the seven OpenASR benchmarks the paper uses, build one manifest per
split (e.g. `ami_test.json`, `librispeech_test_clean.json`, etc.).

## Run Parakeet 5-best

```bash
python ~/NeMo/examples/asr/transcribe_speech.py \
    pretrained_name=nvidia/parakeet-tdt-0.6b-v3 \
    dataset_manifest=ami_test.json \
    output_filename=ami_test_parakeet_5best.json \
    batch_size=16 \
    decoder_type=rnnt \
    rnnt_decoding.strategy=maes \
    rnnt_decoding.beam.beam_size=5 \
    rnnt_decoding.beam.return_best_hypothesis=False
```

Swap `beam_size=5` for `beam_size=1` to produce the 1-best ablation
variant.

## Convert NeMo output to the Speech-Hands layout

NeMo writes each hypothesis group under `pred_text`. Reformat to
`{audio, gold, parakeet_nbest}` lines and drop into
`$SPEECH_HANDS_DATA/asr_nbest/parakeet_5_best/`:

```python
import json, os, sys
SPEECH_HANDS_DATA = os.environ.get("SPEECH_HANDS_DATA", "./data")
split = sys.argv[1]                         # e.g. ami_test
nemo_path = sys.argv[2]                     # NeMo's output_filename

rows = [json.loads(l) for l in open(nemo_path)]
out_dir = f"{SPEECH_HANDS_DATA}/asr_nbest/parakeet_5_best"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/{split}_parakeet_5_best.jsonl", "w") as f:
    for r in rows:
        hyps = r["pred_text"] if isinstance(r["pred_text"], list) else [r["pred_text"]]
        f.write(json.dumps({
            "audio": r["audio_filepath"],
            "gold": r.get("text", ""),
            "parakeet_nbest": hyps,
        }) + "\n")
```

After this step, `scripts/process_asr_nbest.py` (5-best supervision) and
`scripts/process_asr_1best.py` (1-best supervision) consume the jsonl
files directly.
