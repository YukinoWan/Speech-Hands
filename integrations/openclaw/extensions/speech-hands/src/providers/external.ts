/**
 * External provider — delegates to OpenClaw's already-installed skills
 * rather than re-implementing ASR / AudioQA from scratch.
 *
 * For `task === "transcribe"`:
 *   invokes the `openai-whisper` skill (local Whisper CLI via OpenClaw's
 *   skill runtime). This keeps Speech-Hands composable with the rest
 *   of the OpenClaw skill ecosystem.
 *
 * For `task === "qa"`:
 *   POSTs audio + question to the `externalAudioQaUrl` endpoint
 *   configured by the user (e.g. a self-hosted Audio Flamingo 3).
 *   If no URL is set, returns an empty prediction and the pipeline
 *   degenerates to internal-only.
 */

import type { ProcessAudioInput, ResolvedSpeechHandsConfig } from "../types";

export interface ExternalProviderResult {
  /** Top-1 / best prediction text. */
  pred: string;
  /** Full N-best list if available (ASR) or multiple samples (AudioQA). */
  nbest?: string[];
  /** Name of the underlying provider we consulted. */
  provider: string;
}

/**
 * Cross-skill RPC handle supplied by the OpenClaw runtime. Skills can
 * be invoked through the SDK's `invokeSkill()` helper; we accept it as
 * an injected dependency so this file stays testable in isolation.
 */
export interface SkillInvoker {
  (skillName: string, args: Record<string, unknown>): Promise<unknown>;
}

export async function callExternalProvider(
  input: ProcessAudioInput,
  cfg: ResolvedSpeechHandsConfig,
  invokeSkill: SkillInvoker,
): Promise<ExternalProviderResult> {
  if (input.task === "transcribe") {
    return callOpenAIWhisperSkill(input, cfg, invokeSkill);
  }
  return callAudioQaExternal(input, cfg);
}

async function callOpenAIWhisperSkill(
  input: ProcessAudioInput,
  cfg: ResolvedSpeechHandsConfig,
  invokeSkill: SkillInvoker,
): Promise<ExternalProviderResult> {
  const skillName = cfg.externalAsrSkill;
  const whisperResult = (await invokeSkill(skillName, {
    audio: input.audio,
    model: "medium",
    outputFormat: "txt",
  })) as { text?: string; alternatives?: string[] } | string;

  if (typeof whisperResult === "string") {
    return { pred: whisperResult.trim(), provider: skillName };
  }
  return {
    pred: (whisperResult.text ?? "").trim(),
    nbest: whisperResult.alternatives,
    provider: skillName,
  };
}

async function callAudioQaExternal(
  input: ProcessAudioInput,
  cfg: ResolvedSpeechHandsConfig,
): Promise<ExternalProviderResult> {
  if (!cfg.externalAudioQaUrl) {
    return { pred: "", provider: "<none>" };
  }
  const audio =
    typeof input.audio === "string"
      ? await readAudioAsBase64(input.audio)
      : input.audio.toString("base64");

  const samples = cfg.majoritySampling ? 5 : 1;
  const body = {
    audio,
    question: input.question,
    samples,
  };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
  try {
    const res = await fetch(cfg.externalAudioQaUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      throw new Error(
        `speech-hands external audio-qa: HTTP ${res.status} from ${cfg.externalAudioQaUrl}`,
      );
    }
    const j = (await res.json()) as { pred: string; samples?: string[] };
    return { pred: j.pred, nbest: j.samples, provider: "audio-flamingo-3" };
  } finally {
    clearTimeout(timer);
  }
}

async function readAudioAsBase64(filepath: string): Promise<string> {
  const { readFile } = await import("node:fs/promises");
  const buf = await readFile(filepath);
  return buf.toString("base64");
}
