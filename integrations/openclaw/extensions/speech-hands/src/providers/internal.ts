/**
 * Internal provider — calls the user-hosted Speech-Hands inference
 * server that runs the fine-tuned Qwen2.5-Omni-7B.
 *
 * The server exposes `POST /v1/process`:
 *   request:  { audio: base64, task: "transcribe"|"qa", question?: string,
 *               external_pred?: string, external_nbest?: string[] }
 *   response: { action_token, final, internal_pred, routing_confidence? }
 *
 * See `integrations/openclaw/server/` for a reference FastAPI
 * implementation.
 */

import type { ProcessAudioInput, ResolvedSpeechHandsConfig } from "../types";

export interface InternalProviderResponse {
  action_token: "<internal>" | "<external>" | "<rewrite>";
  final: string;
  internal_pred: string;
  routing_confidence?: number;
}

export interface InternalCallContext {
  externalPred?: string;
  externalNbest?: string[];
}

export async function callInternalProvider(
  input: ProcessAudioInput,
  ctx: InternalCallContext,
  cfg: ResolvedSpeechHandsConfig,
): Promise<InternalProviderResponse> {
  const url = new URL("/v1/process", cfg.inferenceServerUrl).toString();
  const audio =
    typeof input.audio === "string"
      ? await readAudioAsBase64(input.audio)
      : input.audio.toString("base64");

  const body = {
    audio,
    task: input.task,
    question: input.question,
    external_pred: ctx.externalPred,
    external_nbest: ctx.externalNbest,
  };

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      throw new Error(
        `speech-hands internal provider: HTTP ${res.status} from ${url}`,
      );
    }
    return (await res.json()) as InternalProviderResponse;
  } finally {
    clearTimeout(timer);
  }
}

async function readAudioAsBase64(filepath: string): Promise<string> {
  const { readFile } = await import("node:fs/promises");
  const buf = await readFile(filepath);
  return buf.toString("base64");
}
