# @openclaw/speech-hands

Voice-input pipeline for OpenClaw. Symmetric with
[`speech-core`](../speech-core) (which handles TTS output); Speech-Hands
handles **audio input** with a self-reflection agent at the core.

## What it does

For every voice-input event, Speech-Hands runs two perception paths
in parallel:

- **Internal** — a fine-tuned Qwen2.5-Omni-7B that makes its own
  prediction directly from the audio.
- **External** — an existing OpenClaw skill (default:
  [`openai-whisper`](../../skills/openai-whisper)) for ASR, or a
  configured HTTP endpoint (e.g. Audio Flamingo 3) for audio QA.

It then emits one of three action tokens (`<internal>`, `<external>`,
`<rewrite>`) and returns a final answer. The paper reports that this
self-reflection layer gives a 12.1% relative WER reduction on seven
OpenASR benchmarks and 77.37% accuracy on DCASE 2025 AudioQA.

## Quick start

```ts
import { processAudio } from "@openclaw/speech-hands";
import { invokeSkill } from "openclaw/plugin-sdk/skills";

// ASR example
const asr = await processAudio(
  { audio: "/tmp/utterance.wav", task: "transcribe" },
  {
    config: {
      inferenceServerUrl: "http://localhost:8080",
      externalAsrSkill: "openai-whisper",
    },
    invokeSkill,
  },
);
console.log(asr.actionToken, asr.final);
// → <external> "now don't burst into a tempest at that"

// Audio-QA example
const qa = await processAudio(
  {
    audio: audioBuffer,
    task: "qa",
    question: "What emotional atmosphere does the music convey?",
  },
  {
    config: {
      inferenceServerUrl: "http://localhost:8080",
      externalAudioQaUrl: "https://your-af3.example.com/v1/qa",
      majoritySampling: true,
    },
    invokeSkill,
  },
);
// qa.actionToken → <external>; qa.final → "D. Sad and reflective"
```

## Configuration

```jsonc
// openclaw.json (workspace-level)
{
  "extensions": {
    "@openclaw/speech-hands": {
      "inferenceServerUrl": "http://localhost:8080",
      "externalAsrSkill": "openai-whisper",
      "externalAudioQaUrl": null,        // optional
      "majoritySampling": true,
      "timeoutMs": 30000
    }
  }
}
```

## Inference server

The internal Qwen2.5-Omni-7B fine-tuned with Speech-Hands supervision
runs in a Python process, not in-process with OpenClaw. Users deploy
it themselves. A reference FastAPI server (Dockerfile +
`requirements.txt`) lives at
[`integrations/openclaw/server/`](../../server).

API contract (`POST /v1/process`):

```jsonc
// request
{
  "audio": "<base64>",
  "task": "transcribe" | "qa",
  "question": "optional — required if task=qa",
  "external_pred": "optional top-1 from the external tool",
  "external_nbest": ["optional", "5-best", "list"]
}

// response
{
  "action_token": "<internal> | <external> | <rewrite>",
  "final": "the final answer",
  "internal_pred": "what Qwen-Omni alone produced",
  "routing_confidence": 0.87
}
```

## Composability

Speech-Hands deliberately reuses existing OpenClaw skills for the
external path. For ASR it calls the `openai-whisper` skill through the
plugin-SDK's `invokeSkill(...)` helper; it does not shell out to
`whisper` directly. This keeps installation requirements unchanged
and ensures Speech-Hands benefits from any future improvements to the
Whisper skill.

## Provenance

- Paper: *Speech-Hands: A Self-Reflection Voice Agentic Approach to
  Speech Recognition and Audio Reasoning with Omni Perception*, ACL
  2026.
- Project page and live demo:
  https://anonymous-paper-page.github.io/Speech-Hands/
- Reference implementation (including the inference server used here):
  https://github.com/Anonymous-paper-page/Speech-Hands
