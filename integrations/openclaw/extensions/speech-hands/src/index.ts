/**
 * Public entry point for the Speech-Hands extension.
 *
 * Consumers — typically the OpenClaw voice-input pipeline or the
 * `skills/speech-hands` skill — import from here:
 *
 *   import { processAudio, type ProcessAudioResult }
 *     from "openclaw/extensions/speech-hands";
 */

export { processAudio, resolveSpeechHandsConfig } from "./pipeline";
export type {
  ActionToken,
  SpeechHandsTask,
  ProcessAudioInput,
  ProcessAudioResult,
  SpeechHandsConfig,
  ResolvedSpeechHandsConfig,
} from "./types";
export type { SkillInvoker } from "./providers/external";
