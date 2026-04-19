"""Reference FastAPI server for the Speech-Hands OpenClaw integration.

Implements the `POST /v1/transcribe` contract that
`@openclaw/speech-hands-provider` speaks to. This is a thin HTTP wrapper
around the Speech-Hands research pipeline — users bring their own
fine-tuned Qwen2.5-Omni-7B checkpoint and (optionally) a local Whisper
install for the external path; both live inside the server process so
the openclaw extension stays a pure HTTP client.

Request shape mirrors openclaw's AudioTranscriptionRequest (just in
JSON form with the buffer base64-encoded, since the openclaw runtime
already handles buffer/mime/language):

  POST /v1/transcribe
  { "audio": "<base64>",
    "file_name": "utterance.wav",
    "mime": "audio/wav",
    "model": "speech-hands-qwen2.5-omni-7b",
    "language": "en" }

Response (extra fields beyond openclaw's `{text, model}` are for debug /
telemetry; the extension discards them):

  { "text": "final transcript",
    "model": "speech-hands-qwen2.5-omni-7b",
    "action_token": "<internal>" | "<external>" | "<rewrite>",
    "internal_pred": "...",
    "external_pred": "...",
    "routing_confidence": 0.87 }
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


class TranscribeRequest(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio bytes.")
    file_name: Optional[str] = Field(None, description='e.g. "utterance.wav"')
    mime: Optional[str] = Field(None, description='e.g. "audio/wav"')
    model: Optional[str] = None
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    model: str
    action_token: Optional[str] = None
    internal_pred: Optional[str] = None
    external_pred: Optional[str] = None
    routing_confidence: Optional[float] = None


@app.on_event("startup")
def _load_pipeline() -> None:
    global pipeline
    pipeline = SpeechHandsPipeline.from_checkpoint(CHECKPOINT_PATH, device=DEVICE)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "checkpoint": CHECKPOINT_PATH, "device": DEVICE}


@app.post("/v1/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    if pipeline is None:
        raise HTTPException(503, "pipeline not loaded")

    try:
        audio_bytes = base64.b64decode(req.audio)
    except Exception as exc:
        raise HTTPException(400, f"invalid base64 audio: {exc}") from exc

    suffix = _suffix_from(req.file_name, req.mime)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as fh:
        fh.write(audio_bytes)
        fh.flush()
        out = pipeline.transcribe(
            audio_path=fh.name,
            language=req.language,
        )

    return TranscribeResponse(
        text=out["final"],
        model=req.model or CHECKPOINT_PATH,
        action_token=out.get("action_token"),
        internal_pred=out.get("internal_pred"),
        external_pred=out.get("external_pred"),
        routing_confidence=out.get("routing_confidence"),
    )


def _suffix_from(file_name: Optional[str], mime: Optional[str]) -> str:
    if file_name and "." in file_name:
        return "." + file_name.rsplit(".", 1)[-1]
    if mime:
        trailing = mime.split("/")[-1]
        if trailing in ("wav", "mpeg", "mp3", "m4a", "flac", "ogg"):
            return "." + ("mp3" if trailing == "mpeg" else trailing)
    return ".wav"
