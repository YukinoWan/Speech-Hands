# Speech-Hands ↔ OpenClaw integration

This directory is the **upstream-ready** OpenClaw integration for
Speech-Hands. Layout mirrors how files need to land in the
[openclaw/openclaw](https://github.com/openclaw/openclaw) monorepo so
that the PR is a straight copy-paste.

```
integrations/openclaw/
├── extensions/speech-hands/   → goes to  openclaw/extensions/speech-hands/
│   └── MediaUnderstandingProvider for self-reflection ASR.
└── server/                    → stays in this repo (not shipped upstream)
    └── Reference FastAPI inference server the extension talks to.
```

## Why a MediaUnderstandingProvider, not a skill?

OpenClaw auto-selects a `MediaUnderstandingProvider` whenever the agent
receives audio — it happens before the LLM sees anything. That is
exactly the surface we want: the agent doesn't need to decide "should I
call Speech-Hands?" at runtime; the runtime does, transparently. So we
register as a provider rather than as a declarative skill.

This matches how
[`@openclaw/deepgram-provider`](https://github.com/openclaw/openclaw/tree/main/extensions/deepgram)
works.

## Why ship an inference server separately?

The internal omni-LLM (fine-tuned Qwen2.5-Omni-7B, ~18 GB fp16) is far
too heavy to load in-process alongside openclaw. The extension is a
thin HTTP client that speaks to a user-hosted Python server. A
reference implementation ships in `server/`; any server honouring the
[`POST /v1/transcribe` contract](extensions/speech-hands/README.md)
works.

## Submitting the PR

```bash
# 1. Fork openclaw/openclaw as Anonymous-paper-page.
# 2. Copy the extension into the fork:
cp -R extensions/speech-hands /path/to/openclaw-fork/extensions/speech-hands
# 3. From the fork root:
pnpm install
pnpm -F @openclaw/speech-hands-provider build
pnpm -F @openclaw/speech-hands-provider test
# 4. Open a **Draft** PR against openclaw/openclaw:main.
```

The PR stays as Draft through the ACL 2026 anonymity period; it moves
to Ready-for-review after paper notification (2026-05).

## Provenance

- Paper: *Speech-Hands: A Self-Reflection Voice Agentic Approach to
  Speech Recognition and Audio Reasoning with Omni Perception*, ACL 2026.
- Project page: https://anonymous-paper-page.github.io/Speech-Hands/
- Reference implementation (this repo): https://github.com/Anonymous-paper-page/Speech-Hands
