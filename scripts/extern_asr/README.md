# External ASR Inference

Reference scripts for running the three external ASR models compared in the
paper (Whisper-v2-large, Canary-1b, Parakeet-TDT-0.6B-v3) and writing their
N-best hypotheses into the format consumed by `scripts/process_asr_*.py`.

| Subdir | Model | Script |
|---|---|---|
| [whisper/](whisper/) | openai/whisper-large-v3 | `whisper_v3_nbest.py` |
| [canary/](canary/) | nvidia/canary-1b | `canary_inference.py` |
| [parakeet/](parakeet/) | nvidia/parakeet-tdt-0.6b-v3 | see `README.md` — minimal NeMo snippet |

## Output layout

All three write into `$SPEECH_HANDS_DATA/asr_nbest/<model>/`, which
`scripts/process_asr_nbest.py` and `scripts/process_asr_1best.py` then read
when building self-reflection supervision data.

```
$SPEECH_HANDS_DATA/asr_nbest/
├── parakeet/            # 1-best (ablation)
├── parakeet_5_best/     # 5-best (main experiment)
├── canary/              # 5-best sampled
└── whisper/             # 5-best beam search
```

## Notes

- Parakeet: paper's main external ASR. Install NeMo (`pip install nemo_toolkit['asr']`).
- Canary: NeMo's EncDecMultiTaskModel — no native beam search, approximated by
  temperature sampling (5 samples at T=0.3).
- Whisper: HuggingFace `transformers` pipeline API, `num_beams=20`,
  `num_return_sequences=5`.

None of these scripts ship the dataset loaders — users should adapt the
`load_split()` helpers (see `canary/canary_inference.py`) to point at their
local copies of the seven OpenASR benchmarks.
