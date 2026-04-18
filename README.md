# Speech-Hands

**A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception**

[![arXiv](https://img.shields.io/badge/arXiv-2601.09413-b31b1b.svg)](https://arxiv.org/abs/2601.09413)
[![ACL 2026](https://img.shields.io/badge/ACL-2026%20Main-4b44ce.svg)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Zhen Wan, Chao-Han Huck Yang, Jinchuan Tian, Hanrong Ye, Ankita Pasad, Szu-wei Fu, Arushi Goel, Ryo Hachiuma, Shizhe Diao, Kunal Dhawan, Sreyan Ghosh, Yusuke Hirota, Zhehuai Chen, Rafael Valle, Ehsan Hosseini Asl, Chenhui Chu, Shinji Watanabe, Yu-Chiang Frank Wang, Boris Ginsburg

<p align="center"><img src="docs/figures/teaser.png" alt="Speech-Hands overview" width="780"></p>

> Speech-Hands acts as a dynamic orchestrator that predicts a special *action token* to govern its cognitive strategy for ASR and multi-domain audio reasoning.

**Demo**: an anonymous static project page lives under [`docs/`](docs/) and is also shipped as a standalone Jupyter notebook at [`demo/demo.ipynb`](demo/demo.ipynb). Both walk through seven representative DCASE 2025 AudioQA dev cases with real audio, the internal vs external predictions, and the self-reflection routing decision. See the [Demo](#demo) section below for how to view them.

> **Checkpoints are not included in this release.** The demo showcases the learned self-reflection behavior on recorded predictions from the paper's runs, so no fine-tuned weights are required to inspect the method. Users who want to reproduce full numbers can re-run the training pipeline from the provided code + configs.

---

## TL;DR

Jointly fine-tuning a voice-LLM on speech recognition and audio reasoning often *hurts* performance, because the model is misled by noisy hypotheses coming from external perception models. **Speech-Hands** re-casts the problem as an explicit **self-reflection decision**: for each example the model emits one of three learnable tokens —

| Token | Meaning |
|---|---|
| `<internal>` | Trust my own prediction, ignore external hypotheses |
| `<external>` | Defer to the external model's top candidate |
| `<rewrite>`  | Neither is right; re-derive the answer from audio + candidates |

On 7 OpenASR benchmarks we obtain **12.1% WER relative improvement** over baselines, and **77.37%** on DCASE 2025 AudioQA.

<p align="center"><img src="docs/figures/method.png" alt="Speech-Hands framework" width="860"></p>

> *Framework overview — a special action token is generated at the beginning to decide whether to use audio perception (transcription hypotheses / caption) or not.*

---

## Headline Results

**ASR** — WER (%) across 7 OpenASR benchmarks. Speech-Hands pairs with each of the three external ASR models we study (Whisper-v2-large, Canary-1b-v2, Parakeet-TDT-0.6B-v3); every pairing beats the corresponding single ASR baseline *and* the cascaded GER variant.

| Model | AMI | Tedlium | GigaSpeech | SPGISpeech | VoxPopuli | Libri-clean | Libri-other | **avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Whisper-v2-large | 16.88 | 4.32 | 11.45 | 3.94 | 7.57 | 2.91 | 5.15 | 7.17 |
| Canary-1b-v2 | 19.80 | 4.78 | 11.66 | 3.08 | 6.35 | 1.73 | 3.17 | 7.22 |
| Parakeet-TDT-0.6B-v3 | 12.69 | 4.90 | 12.24 | 3.16 | 6.48 | 1.89 | 3.37 | 6.68 |
| Qwen2.5-Omni (base) | 19.77 | 5.17 | 11.26 | 4.58 | 6.59 | 2.09 | 3.85 | 7.33 |
| GER ⇒ Whisper-v2 (cascaded) | 23.44 | 6.15 | 12.15 | 3.94 | 7.53 | 2.97 | 4.89 | 8.44 |
| GER ⇒ Canary (cascaded) | 24.58 | 6.38 | 12.43 | 4.02 | 7.72 | 3.05 | 5.01 | 8.74 |
| GER ⇒ Parakeet (cascaded) | 22.91 | 6.09 | 12.10 | 3.98 | 7.49 | 2.92 | 4.84 | 8.33 |
| Speech-Hands ⇌ Whisper-v2 | 15.03 | 4.45 | 12.37 | 3.01 | 6.49 | 1.86 | 3.46 | 6.67 |
| Speech-Hands ⇌ Canary | 15.29 | 4.21 | 10.87 | **2.17** | 5.96 | **1.61** | **3.07** | 6.17 |
| **Speech-Hands ⇌ Parakeet** | **11.20** | 4.37 | 11.10 | 2.26 | **6.02** | 1.67 | 3.18 | **5.69** |

**AudioQA** — Accuracy (%) on DCASE 2025 (Speech-Hands ⇌ *Audio Flamingo 3* vs. strongest baselines):

| Model / Setting | Bio-acoustic | Soundscape | Complex QA | **avg.** |
|---|---:|---:|---:|---:|
| Qwen2.5-Omni (base) | 47.32 | 56.32 | 59.89 | 57.87 |
| Audio Flamingo 3 | 71.88 | 57.31 | 81.26 | 74.49 |
| GER ⇒ AF3 (cascaded) | 76.29 | 52.02 | 77.48 | 68.93 |
| Speech-Hands — SFT | 67.86 | 58.29 | 83.34 | 75.75 |
| **Speech-Hands — SFT + majority sampling** | **81.25** | **59.40** | **85.70** | **77.37** |

Full per-benchmark tables, cascaded baselines (GER ⇒ Whisper / Canary), ablations, and action-token F1 analysis live in the paper and on the project page ([docs/index.html](docs/index.html)).

---

## Repository Layout

```
Speech-Hands/
├── configs/                     # Training YAMLs (LLaMA-Factory format)
│   ├── asr_main/                # Main ASR experiments (Parakeet 5-best + self-reflection)
│   ├── asr_ablation_1best/      # Ablation: 1-best vs 5-best
│   ├── asr_ablation_ger/        # Ablation: vs GER baseline
│   ├── audio_qa/                # DCASE 2025 AudioQA (1-best external)
│   ├── deepspeed/               # DeepSpeed Zero configs
│   ├── accelerate/              # FSDP configs
│   └── preliminary/             # Earlier exploration (not reported in paper)
├── scripts/
│   ├── train/                   # Shell wrappers per experiment
│   ├── eval/                    # Evaluation wrappers
│   ├── compute_wer.py           # WER computation
│   ├── compute_acc.py           # QA accuracy + confusion matrices
│   ├── process_asr_{n,1}best.py # ASR supervision builders
│   ├── process_2025_{n,1}best.py# AudioQA supervision builders
│   ├── rag_prompt.py            # Self-reflection prompt templates
│   └── qwen_omni_merge.py       # Merge LoRA/full ckpt for inference
├── test_qwen_{rag,2025,ger,...}.py  # Inference entry points
├── {internal,external,rewrite}_data.json  # Self-reflection SFT labels
├── src/llamafactory/            # Training framework (LLaMA Factory, Apache 2.0)
└── llava/                       # Multimodal encoders (adapted from NVIDIA VILA, Apache 2.0)
```

Each training config has a matching shell wrapper under `scripts/train/` that calls `llamafactory-cli train <yaml>` plus a merge + eval chain.

---

## Demo

Two ways to explore Speech-Hands' self-reflection behavior without any training or inference compute:

### 1. Project web page (static, no setup)

The [`docs/`](docs/) directory is a self-contained HTML site that renders seven DCASE 2025 AudioQA cases (3 `<internal>`, 3 `<external>`, 1 `<rewrite>`) with inline audio players and a side-by-side comparison table. Open it locally:

```bash
# quick open
open docs/index.html                          # macOS
xdg-open docs/index.html                      # Linux

# or serve on localhost (audio seeks more smoothly this way):
python3 -m http.server 8765 --directory docs
# then browse http://localhost:8765/
```

The same site is meant to be deployed as an anonymous GitHub Pages project page: Settings → Pages → source = *main / `/docs`*.

### 2. Jupyter notebook

`demo/demo.ipynb` is a standalone notebook version of the same cases. Useful if you prefer cell-based exploration or want to extend the case list.

```bash
jupyter notebook demo/demo.ipynb
```

Both views read from `demo/demo_cases.json` / `docs/cases.json`, which bundle the 9-row supervision labels + question text so that no checkpoint / external output files are needed at runtime.

---

## Setup

```bash
git clone https://github.com/<user>/Speech-Hands.git
cd Speech-Hands

# Create an env (conda shown; venv works too)
conda create -n speech-hands python=3.10 -y
conda activate speech-hands

pip install -e .
pip install flash-attn --no-build-isolation    # optional: 2-3x faster training
```

This installs the vendored `llamafactory` training framework along with the dependencies in [`requirements.txt`](requirements.txt) (transformers, peft, trl, deepspeed, librosa, etc.).

---

## Data Preparation

Speech-Hands consumes three data sources. Arrange them under `./data/` (or point `SPEECH_HANDS_DATA` elsewhere — see [Environment Variables](#environment-variables) below):

```
data/
├── asr_nbest/                   # External ASR N-best hypotheses
│   ├── parakeet/                # parakeet-tdt-0.6b-v3
│   ├── parakeet_5_best/         # 5-best variant used by the main experiment
│   ├── whisper/                 # whisper-v2-large
│   └── canary/                  # canary-1b-v2
│       └── {dataset}_{split}_{asr}_5_best.jsonl
├── dcase_audioqa/               # DCASE 2025 AudioQA official release
│   ├── 2025_{train,dev}_sharegpt.json
│   ├── 2025_{train,dev}_sharegpt_rag_{ckpt}.json   # generated by process_2025_1best.py
│   └── audio/
│       ├── train/audio_XXXXX.wav
│       └── dev/audio_XXXXX.wav
└── sharegpt/                    # ShareGPT-formatted RAG prompts (built by scripts/rag_prompt.py)
    └── {asr}_prompt_v2/{dataset}_{asr}_5_best_with_audio.json
```

The seven ASR benchmarks are AMI, TEDLIUM, GigaSpeech, SPGISpeech, VoxPopuli, LibriSpeech-clean, LibriSpeech-other (OpenASR leaderboard). See each respective dataset's license page for acquisition.

### Building supervision data

The self-reflection labels are computed from (a) gold transcripts/answers, (b) internal model predictions, (c) external model predictions. Once you have the raw data above, run:

```bash
# ASR track: build 5-best RAG supervision
python scripts/process_asr_nbest.py

# ASR track: build 1-best ablation supervision
python scripts/process_asr_1best.py

# AudioQA track: build 1-best RAG supervision
python scripts/process_2025_1best.py <subset> <ckpt>
```

These produce SFT JSON files under `data/sharegpt/...` and `data/dcase_audioqa/2025_*_sharegpt_rag_*.json`. Three ready-made label files are shipped at the repo root:

| File | Routing gold |
|---|---|
| `internal_data.json` | `<internal>` — audio samples where the internal model is correct |
| `external_data.json` | `<external>` — audio samples where only the external model is correct |
| `rewrite_data.json`  | `<rewrite>` — audio samples where both internal and external are wrong |

---

## Environment Variables

All scripts read their paths from three optional env vars (defaults are repo-relative):

| Variable | Default | Used for |
|---|---|---|
| `SPEECH_HANDS_DATA` | `./data` | ASR N-best, DCASE AudioQA, ShareGPT prompts |
| `SPEECH_HANDS_OUTPUT` | `./output` | Baseline / Flamingo inference outputs |
| `SPEECH_HANDS_CKPT` | `./target_dir` | Merged inference checkpoints |

```bash
export SPEECH_HANDS_DATA=/path/to/your/data
export SPEECH_HANDS_OUTPUT=/scratch/output
export SPEECH_HANDS_CKPT=/scratch/target_dir
```

---

## Training

All training uses [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) under the hood.

### ASR — main experiment (Parakeet 5-best + self-reflection RAG)

```bash
# Per-dataset training (main setting reported in the paper)
bash scripts/train/asr_main/run_ami_with_audio.sh
bash scripts/train/asr_main/run_librispeech_with_audio.sh
bash scripts/train/asr_main/run_tedlium_with_audio.sh
bash scripts/train/asr_main/run_gigaspeech_with_audio.sh
bash scripts/train/asr_main/run_spgispeech_with_audio.sh
bash scripts/train/asr_main/run_voxpopuli_with_audio.sh
# Optional: joint training across all datasets
bash scripts/train/asr_main/run_all_with_audio.sh
```

To swap external ASR models, edit the `dataset:` field in the corresponding YAML:

```yaml
# configs/asr_main/qwen2_5omni_full_sft_ami_with_audio_rag.yaml
dataset: ami-parakeet-5-best-train-with-audio-v3    # Parakeet (paper main)
# dataset: ami-whisper-5-best-train-with-audio      # Whisper variant
# dataset: ami-canary-5-best-train                  # Canary variant
```

### AudioQA — DCASE 2025 (1-best external)

```bash
bash scripts/train/audio_qa/run_all_with_audio_rag_v6.sh
```

### Ablations

```bash
bash scripts/train/asr_ablation_1best/run_<dataset>_with_audio.sh     # 1-best vs 5-best
bash scripts/train/asr_ablation_ger/run_<dataset>_with_audio.sh       # vs GER baseline
```

### Wrapping for your launcher

The shell wrappers are plain `bash` scripts. Wrap them in your own launcher as needed:

```bash
sbatch --wrap='bash scripts/train/asr_main/run_ami_with_audio.sh'
# or
torchrun --nproc_per_node=8 -m llamafactory.launcher ...
```

---

## Evaluation

### WER (ASR)

```bash
# Compare Speech-Hands against baseline + external model
python scripts/compute_wer.py <task>         # e.g. ami_test, librispeech_test_clean

# Per-sample breakdown
python scripts/analyze_wer.py <task> <ckpt>
python wer_best_of_two.py <task> <ckpt>
```

### QA accuracy (AudioQA)

```bash
python scripts/compute_acc.py <ckpt>         # computes routing confusion matrices too
```

### BLEU / ROUGE (Understanding track)

```bash
python scripts/eval_bleu_rouge.py <predictions.json> <references.json>
```

### Inference entry points

```bash
python test_qwen_rag.py   <task> <ckpt> [asr]    # Speech-Hands on an ASR task
python test_qwen_2025.py  <ckpt>                 # Speech-Hands on DCASE 2025
python test_qwen_ger.py   <task> <ckpt> <no_audio>   # GER baseline
python test_flamingo_2025.py <split>             # Audio Flamingo baseline
python test_qwen_baseline.py <task> <no_audio>   # Qwen2.5-Omni baseline
```

---

## Citation

Once the ACL Anthology ID is live we will update this. For now:

```bibtex
@article{wan2026speechhands,
  title   = {Speech-Hands: A Self-Reflection Voice Agentic Approach to
             Speech Recognition and Audio Reasoning with Omni Perception},
  author  = {Wan, Zhen and Yang, Chao-Han Huck and Tian, Jinchuan and Ye, Hanrong
             and Pasad, Ankita and Fu, Szu-wei and Goel, Arushi and Hachiuma, Ryo
             and Diao, Shizhe and Dhawan, Kunal and Ghosh, Sreyan and Hirota, Yusuke
             and Chen, Zhehuai and Valle, Rafael and Hosseini Asl, Ehsan
             and Chu, Chenhui and Watanabe, Shinji and Wang, Yu-Chiang Frank
             and Ginsburg, Boris},
  journal = {arXiv preprint arXiv:2601.09413},
  year    = {2026}
}
```

---

## Acknowledgments

- Training framework built on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) (Apache 2.0).
- Multimodal `llava/` module adapted from [NVIDIA VILA](https://github.com/NVlabs/VILA) (Apache 2.0).
- Backbone: [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B).
- External ASR: [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [Whisper-v2-large](https://huggingface.co/openai/whisper-large-v2), [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b).
- External audio reasoning: [Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
