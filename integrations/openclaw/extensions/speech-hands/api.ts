/**
 * Public plugin-SDK surface for the Speech-Hands extension. Kept
 * symmetric with `extensions/speech-core/api.ts`, which re-exports
 * from a pre-registered SDK module.
 *
 * Because Speech-Hands is a community contribution rather than a
 * bundled OpenClaw primitive, we export the types inline here rather
 * than depending on `openclaw/plugin-sdk/speech-hands` being declared
 * upstream. If this extension is later promoted into the core SDK,
 * flip this file to `export * from "openclaw/plugin-sdk/speech-hands";`.
 */
export type {
  ActionToken,
  SpeechHandsTask,
  ProcessAudioInput,
  ProcessAudioResult,
  SpeechHandsConfig,
  ResolvedSpeechHandsConfig,
} from "./src/types";
