---
name: speech-hands
description: Self-reflection voice-input agent that fuses an internal omni-LLM with an external perception tool (e.g. Whisper) for higher-accuracy ASR and AudioQA.
homepage: https://anonymous-paper-page.github.io/Speech-Hands/
metadata:
  {
    "openclaw":
      {
        "emoji": "🤝",
        "requires":
          {
            "extensions": ["@openclaw/speech-hands"],
            "skills": ["openai-whisper"],
            "services": ["speech-hands-server"],
          },
        "install":
          [
            {
              "id": "pnpm",
              "kind": "pnpm",
              "package": "@openclaw/speech-hands",
              "label": "Install Speech-Hands extension",
            },
          ],
      },
  }
---

# Speech-Hands (voice-input core)

Use Speech-Hands when the user gives **audio input** and you want the most
reliable transcript or audio-question answer OpenClaw can produce. It is
symmetric with `speech-core` (TTS): `speech-core` is the voice **output**
core, `speech-hands` is the voice **input** core.

## When to invoke

Prefer `speech-hands` over plain `openai-whisper` when any of the following
hold:

- The audio is noisy, accented, domain-specific (meetings, legal, medical),
  or otherwise hard — Whisper alone tends to hallucinate or drop tokens.
- The user asked a **question about non-speech audio** (music, ambience,
  sound events). Plain Whisper cannot do this at all; Speech-Hands routes
  to Audio Flamingo 3 or a comparable audio-QA model.
- Accuracy matters more than latency (Speech-Hands runs two perception
  paths in parallel plus a self-reflection step, so it is ~1.5–2× slower
  than Whisper alone).

Fall back to plain `openai-whisper` when:

- The Speech-Hands server is not reachable.
- The user explicitly asks for Whisper, or for a fast/cheap transcript.

## Quick start

Transcribe (ASR):

```ts
import { processAudio } from "@openclaw/speech-hands";
import { invokeSkill } from "openclaw/plugin-sdk/skills";

const result = await processAudio(
  { audio: "/tmp/utterance.wav", task: "transcribe" },
  { config: { inferenceServerUrl: "http://localhost:8080" }, invokeSkill },
);
// result.actionToken ∈ { "<internal>", "<external>", "<rewrite>" }
// result.final         — the final transcript
```

Answer a question about audio (AudioQA):

```ts
const qa = await processAudio(
  {
    audio: "/tmp/scene.wav",
    task: "qa",
    question: "What emotional atmosphere does the music convey?",
  },
  {
    config: {
      inferenceServerUrl: "http://localhost:8080",
      externalAudioQaUrl: "https://your-af3.example.com/v1/qa",
    },
    invokeSkill,
  },
);
```

## How it works (short version)

For every request the pipeline runs two paths in parallel and then reflects:

- **Internal** — a fine-tuned Qwen2.5-Omni-7B predicts directly from the
  audio (runs inside the Speech-Hands server).
- **External** — an existing OpenClaw skill (default: `openai-whisper`)
  for ASR, or a configured HTTP endpoint for AudioQA.

The server emits one of three action tokens and returns a final answer:

- `<internal>` — keep the internal prediction.
- `<external>` — defer to the external tool.
- `<rewrite>` — neither is right on its own; synthesize a corrected answer.

Reported gains: 12.1% relative WER reduction on seven OpenASR benchmarks
and 77.37% accuracy on DCASE 2025 AudioQA.

## Configuration

Workspace `openclaw.json`:

```jsonc
{
  "extensions": {
    "@openclaw/speech-hands": {
      "inferenceServerUrl": "http://localhost:8080",
      "externalAsrSkill": "openai-whisper",
      "externalAudioQaUrl": null,
      "majoritySampling": true,
      "timeoutMs": 30000
    }
  }
}
```

## Notes

- The internal omni-LLM runs in a separate Python process. A reference
  FastAPI server (Dockerfile + `requirements.txt`) ships at
  `integrations/openclaw/server/` in the Speech-Hands repo.
- The external ASR path reuses the `openai-whisper` skill via the
  plugin-SDK's `invokeSkill`. You do not need to install anything extra
  for ASR beyond what `openai-whisper` already requires.
- For AudioQA, point `externalAudioQaUrl` at any endpoint that accepts
  `{audio, question}` and returns `{answer}`. Audio Flamingo 3 works
  out of the box.
- Paper: *Speech-Hands: A Self-Reflection Voice Agentic Approach to
  Speech Recognition and Audio Reasoning with Omni Perception*, ACL 2026.
