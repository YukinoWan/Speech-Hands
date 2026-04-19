# Speech-Hands ↔ OpenClaw integration

This directory is the **upstream-ready** OpenClaw integration for
Speech-Hands. It is laid out exactly the way the files need to land in
the [openclaw/openclaw](https://github.com/openclaw/openclaw) monorepo
when we open the PR, so that review is a straight copy-paste.

```
integrations/openclaw/
├── extensions/speech-hands/   → goes to  openclaw/extensions/speech-hands/
│   └── TypeScript package: the processAudio() pipeline.
├── skills/speech-hands/       → goes to  openclaw/skills/speech-hands/
│   └── Declarative SKILL.md: tells the OpenClaw agent when to call us.
└── server/                    → stays in this repo (not shipped upstream)
    └── Reference FastAPI inference server the extension talks to.
```

## Why two pieces?

OpenClaw has two extension points, and Speech-Hands uses both:

- **Skills** (`skills/*/SKILL.md`) are declarative cards the agent reads
  to decide *when and why* to invoke something. They have no code.
- **Extensions** (`extensions/*`) are TypeScript packages that expose
  real functions (`processAudio`, etc.) callable from other extensions
  or from the agent runtime.

`speech-core` (the voice **output** core) follows the same pattern:
`skills/speech-core/SKILL.md` plus `extensions/speech-core/`.
Speech-Hands is the symmetric voice **input** core.

## Why ship an inference server separately?

The internal omni-LLM (fine-tuned Qwen2.5-Omni-7B) is far too heavy to
load in-process alongside OpenClaw — it wants a GPU and ~18 GB of VRAM.
So the extension speaks HTTP to a separately-deployed Python server.
A reference implementation lives in `server/`; users can also swap in
their own as long as it honours the `POST /v1/process` contract
documented in [`extensions/speech-hands/README.md`](extensions/speech-hands/README.md#inference-server).

## Submitting the PR

```bash
# 1. Fork openclaw/openclaw.
# 2. Copy the two pieces into the fork:
cp -R extensions/speech-hands  /path/to/openclaw-fork/extensions/speech-hands
cp -R skills/speech-hands       /path/to/openclaw-fork/skills/speech-hands
# 3. Register @openclaw/speech-hands in the workspace's pnpm catalog if
#    needed, run `pnpm install` from the repo root.
# 4. Open the PR against openclaw/openclaw:main, linking this README and
#    the Speech-Hands paper / project page.
```

## Provenance

- Paper: *Speech-Hands: A Self-Reflection Voice Agentic Approach to
  Speech Recognition and Audio Reasoning with Omni Perception*, ACL 2026.
- Project page: https://anonymous-paper-page.github.io/Speech-Hands/
- Reference implementation (this repo): https://github.com/Anonymous-paper-page/Speech-Hands
