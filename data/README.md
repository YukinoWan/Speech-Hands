# `data/` — dataset registry and supervision files

This directory is Speech-Hands' runtime data root. It is **partially gitignored**
(see [`.gitignore`](../.gitignore)):

| Path | In git? | What it is |
|---|---|---|
| `dataset_info.json` | ✅ yes (whitelisted) | LLaMA-Factory dataset registry — maps every `dataset:` name used in `configs/**/*.yaml` to a physical file under `data/`. |
| `README.md` (this file) | ✅ yes (whitelisted) | — |
| Everything else | ❌ ignored | Large ASR / AudioQA corpora, ShareGPT-format supervision, audio blobs, etc. You populate these locally; they never enter git. |

## Expected layout after running the data-prep pipeline

```
data/
├── dataset_info.json
├── README.md
├── asr_nbest/                     # produced by scripts/extern_asr/**
│   ├── parakeet_5_best/           # 5-best from Parakeet-TDT-0.6B-v3
│   ├── parakeet/                  # 1-best variant
│   ├── whisper/                   # 5-best from Whisper-v2-large
│   └── canary/                    # 5-best from Canary-1b-v2
├── dcase_audioqa/                 # DCASE 2025 AudioQA (from the challenge)
│   ├── 2025_train_sharegpt.json
│   ├── 2025_train_sharegpt_rag_1best_v6.json
│   └── audio/{train,dev}/audio_*.wav
├── sharegpt/                      # ShareGPT-format SFT source (gold)
│   └── whisper_5_best/{d}_{split}_whisper_5_best_{with,no}_audio.json
├── sharegpt_data_parakeet_5_best/ # produced by scripts/process_asr_nbest.py (v3/main)
├── sharegpt_data_parakeet_ger_rag/# produced by scripts/process_asr_nbest.py (GER-RAG ablation)
├── sharegpt_data_parakeet_1best/  # produced by scripts/process_asr_1best.py
└── sharegpt_data_canary_5_best/   # Canary 5-best (only ami registered in the paper)
```

## Using `dataset_info.json`

`dataset_info.json` is consumed by LLaMA Factory's data loader when a training
YAML references a dataset name. Every entry follows:

```json
"<dataset-name>": {
  "file_name": "<path relative to data/>",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations", "audios": "audios" },
  "tags":    { "role_tag": "from", "content_tag": "value",
               "user_tag": "human", "assistant_tag": "gpt" }
}
```

The shipped registry covers **every `dataset:` name used in `configs/**/*.yaml`**.
If you build the ShareGPT-format files with a different directory convention,
edit `file_name` accordingly.

## Checking the registry

```bash
# List every dataset name this repo's configs reference
grep -rhE "^dataset:" configs/**/*.yaml | sed 's/dataset: //' | tr ',' '\n' | sort -u

# All of the above should resolve to an entry in data/dataset_info.json
python3 -c "import json; keys=set(json.load(open('data/dataset_info.json'))); print(len(keys),'entries')"
```
