# Speech-Hands inference server (reference)

Thin FastAPI wrapper that exposes the research-code Speech-Hands
pipeline over HTTP so the OpenClaw extension (`@openclaw/speech-hands`)
can call it. Users are expected to run this themselves — it is **not**
bundled into OpenClaw.

## Prerequisites

- A GPU with enough memory to load the fine-tuned Qwen2.5-Omni-7B
  checkpoint (≈18 GB fp16).
- The Speech-Hands research package (`pip install -e .` from the repo
  root) so that `speech_hands.inference.SpeechHandsPipeline` is
  importable.
- The fine-tuned checkpoint at `SPEECH_HANDS_CHECKPOINT` (default
  `/models/speech-hands-qwen2.5-omni-7b`). See the top-level repo
  README for training recipes.

## Run locally (Python)

```bash
pip install -r requirements.txt
pip install -e /path/to/Speech-Hands       # the research package

export SPEECH_HANDS_CHECKPOINT=/path/to/checkpoint
export SPEECH_HANDS_DEVICE=cuda
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Run with Docker

```bash
docker build -t speech-hands-server .
docker run --gpus all \
  -p 8080:8080 \
  -v /path/to/checkpoint:/models/speech-hands-qwen2.5-omni-7b:ro \
  -v /path/to/Speech-Hands:/opt/speech-hands:ro \
  -e PYTHONPATH=/opt/speech-hands \
  speech-hands-server
```

(The Dockerfile does not bake in the research package because checkpoint
location and commit pinning are deployment-specific. Mount both at run
time.)

## API

`GET /healthz` — liveness probe. Returns the checkpoint path and device.

`POST /v1/process` — main entry point. See the JSON contract in the
extension's [README](../extensions/speech-hands/README.md#inference-server).

## Environment variables

| Variable                  | Default                                    | Purpose                      |
| ------------------------- | ------------------------------------------ | ---------------------------- |
| `SPEECH_HANDS_CHECKPOINT` | `/models/speech-hands-qwen2.5-omni-7b`     | Path to fine-tuned weights.  |
| `SPEECH_HANDS_DEVICE`     | `cuda`                                     | Torch device string.         |
