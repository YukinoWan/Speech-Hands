"""Reference FastAPI server for the Speech-Hands OpenClaw integration.

Implements the `POST /v1/process` contract that `@openclaw/speech-hands`
speaks to. This is a thin wrapper around the Speech-Hands inference code
in the top-level repository — users are expected to bring their own
fine-tuned Qwen2.5-Omni-7B checkpoint (see the repo README for training
recipes and the DCASE / OpenASR data layout).

The internal model, external-evidence fusion, and action-token emission
all live in `speech_hands.inference` (the vendored research code); this
file only exposes them over HTTP.
"""
from __future__ import annotations

import base64
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from speech_hands.inference import SpeechHandsPipeline
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "speech_hands.inference is not importable. Install the Speech-Hands "
        "package from the top-level repo (pip install -e .) before running "
        "this server."
    ) from exc


CHECKPOINT_PATH = os.environ.get(
    "SPEECH_HANDS_CHECKPOINT",
    "/models/speech-hands-qwen2.5-omni-7b",
)
DEVICE = os.environ.get("SPEECH_HANDS_DEVICE", "cuda")

app = FastAPI(title="Speech-Hands Inference Server", version="0.1.0")
pipeline: Optional[SpeechHandsPipeline] = None


class ProcessRequest(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio (wav/mp3/m4a).")
    task: str = Field(..., description='"transcribe" or "qa".')
    question: Optional[str] = None
    external_pred: Optional[str] = None
    external_nbest: Optional[list[str]] = None


class ProcessResponse(BaseModel):
    action_token: str
    final: str
    internal_pred: str
    routing_confidence: Optional[float] = None


@app.on_event("startup")
def _load_pipeline() -> None:
    global pipeline
    pipeline = SpeechHandsPipeline.from_checkpoint(CHECKPOINT_PATH, device=DEVICE)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "checkpoint": CHECKPOINT_PATH, "device": DEVICE}


@app.post("/v1/process", response_model=ProcessResponse)
def process(req: ProcessRequest) -> ProcessResponse:
    if pipeline is None:
        raise HTTPException(503, "pipeline not loaded")
    if req.task not in ("transcribe", "qa"):
        raise HTTPException(400, f"unsupported task: {req.task}")
    if req.task == "qa" and not req.question:
        raise HTTPException(400, "task=qa requires a question")

    try:
        audio_bytes = base64.b64decode(req.audio)
    except Exception as exc:
        raise HTTPException(400, f"invalid base64 audio: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fh:
        fh.write(audio_bytes)
        fh.flush()
        out = pipeline.process(
            audio_path=fh.name,
            task=req.task,
            question=req.question,
            external_pred=req.external_pred,
            external_nbest=req.external_nbest,
        )

    return ProcessResponse(
        action_token=out["action_token"],
        final=out["final"],
        internal_pred=out["internal_pred"],
        routing_confidence=out.get("routing_confidence"),
    )
