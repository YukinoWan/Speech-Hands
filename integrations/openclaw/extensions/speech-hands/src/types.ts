/**
 * Speech-Hands extension — type contract.
 *
 * Speech-Hands is OpenClaw's voice-input pipeline: the symmetric
 * counterpart to `speech-core` (voice output / TTS).
 *
 * On every voice-input event, the extension runs two perception paths
 * in parallel — the internal omni-model (Qwen2.5-Omni-7B, fine-tuned
 * with Speech-Hands supervision) and an external perception tool
 * (e.g. the `openai-whisper` skill, or a hosted Audio Flamingo
 * endpoint) — then emits one of three *action tokens* and returns the
 * final answer.
 */

export type ActionToken = "<internal>" | "<external>" | "<rewrite>";

export type SpeechHandsTask = "transcribe" | "qa";

export interface ProcessAudioInput {
  /** Raw audio bytes (16kHz mono PCM recommended) or a local filepath. */
  audio: Buffer | string;
  /** Task mode. `transcribe` = ASR; `qa` = audio question answering. */
  task: SpeechHandsTask;
  /** Required when `task === "qa"`. Free-text question about the audio. */
  question?: string;
  /** Optional hint to bias external provider choice. */
  externalProvider?: "openai-whisper" | "audio-flamingo-3" | string;
}

export interface ProcessAudioResult {
  /** The action token the agent emitted. */
  actionToken: ActionToken;
  /** Final answer — transcript for ASR, answer for QA. */
  final: string;
  /** What the internal Qwen2.5-Omni path alone produced. Useful for debug. */
  internalPred: string;
  /** What the external perception tool produced. */
  externalPred: string;
  /** Model name of the external perception tool actually consulted. */
  externalProvider: string;
  /**
   * Majority-sampling confidence when applicable (paper's best AudioQA
   * setting samples the external model 5 times and picks the mode).
   */
  confidence?: number;
}

export interface SpeechHandsConfig {
  /**
   * HTTP base URL of the Speech-Hands inference server that hosts the
   * fine-tuned Qwen2.5-Omni model. Users deploy this themselves — see
   * the reference FastAPI server under `integrations/openclaw/server/`.
   */
  inferenceServerUrl: string;
  /**
   * Name of the already-installed OpenClaw skill to invoke as the
   * external ASR tool when `task === "transcribe"`. Defaults to
   * `openai-whisper` to reuse OpenClaw's existing local-Whisper
   * wrapper rather than re-implementing ASR.
   */
  externalAsrSkill?: string;
  /**
   * HTTP endpoint of the external AudioQA perception model (e.g. a
   * hosted Audio Flamingo 3). Speech-Hands calls this when
   * `task === "qa"`. If null, external AudioQA is skipped and the
   * internal prediction is always trusted.
   */
  externalAudioQaUrl?: string | null;
  /**
   * Enable majority sampling of the external AudioQA model (paper's
   * best setting for AudioQA accuracy: 77.37% vs 75.75% without).
   */
  majoritySampling?: boolean;
  /** Request timeout in ms for the inference server. Default 30000. */
  timeoutMs?: number;
}

export interface ResolvedSpeechHandsConfig
  extends Required<Omit<SpeechHandsConfig, "externalAudioQaUrl">> {
  externalAudioQaUrl: string | null;
}
