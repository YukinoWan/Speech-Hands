/**
 * Speech-Hands voice-input pipeline.
 *
 * This is the entry point the OpenClaw runtime calls when audio
 * arrives from the voice-wake / mic surface, before the LLM sees the
 * utterance. Sequence:
 *
 *   1. Kick off external perception (existing OpenClaw skills or
 *      configured HTTP endpoint) in parallel with internal perception.
 *   2. Pass the external top-1 / N-best into the internal Speech-Hands
 *      server as grounding context.
 *   3. The server returns an action token + final answer. We surface
 *      both to the OpenClaw agent loop.
 */

import type {
  ProcessAudioInput,
  ProcessAudioResult,
  ResolvedSpeechHandsConfig,
  SpeechHandsConfig,
} from "./types";
import { callInternalProvider } from "./providers/internal";
import { callExternalProvider, type SkillInvoker } from "./providers/external";

const DEFAULT_CFG: ResolvedSpeechHandsConfig = {
  inferenceServerUrl: "http://localhost:8080",
  externalAsrSkill: "openai-whisper",
  externalAudioQaUrl: null,
  majoritySampling: true,
  timeoutMs: 30_000,
};

export function resolveSpeechHandsConfig(
  partial: SpeechHandsConfig | undefined,
): ResolvedSpeechHandsConfig {
  const out: ResolvedSpeechHandsConfig = { ...DEFAULT_CFG };
  if (!partial) return out;
  if (partial.inferenceServerUrl) out.inferenceServerUrl = partial.inferenceServerUrl;
  if (partial.externalAsrSkill) out.externalAsrSkill = partial.externalAsrSkill;
  if (partial.externalAudioQaUrl !== undefined)
    out.externalAudioQaUrl = partial.externalAudioQaUrl;
  if (partial.majoritySampling !== undefined)
    out.majoritySampling = partial.majoritySampling;
  if (partial.timeoutMs !== undefined) out.timeoutMs = partial.timeoutMs;
  return out;
}

export async function processAudio(
  input: ProcessAudioInput,
  opts: {
    config?: SpeechHandsConfig;
    invokeSkill: SkillInvoker;
  },
): Promise<ProcessAudioResult> {
  if (input.task === "qa" && !input.question) {
    throw new Error("speech-hands: task=qa requires `question`");
  }
  const cfg = resolveSpeechHandsConfig(opts.config);

  const external = await callExternalProvider(input, cfg, opts.invokeSkill);
  const internal = await callInternalProvider(
    input,
    { externalPred: external.pred, externalNbest: external.nbest },
    cfg,
  );

  return {
    actionToken: internal.action_token,
    final: internal.final,
    internalPred: internal.internal_pred,
    externalPred: external.pred,
    externalProvider: external.provider,
    confidence: internal.routing_confidence,
  };
}
