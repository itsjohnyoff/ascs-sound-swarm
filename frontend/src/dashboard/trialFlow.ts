import type { Detection } from "../shared/types";

export type TrialPhase = "idle" | "chirping" | "listening" | "review";

/** A detection belongs to the current trial only if it is new, first-emission, and after the trial started. */
export function isCandidate(d: Detection, startedAtS: number): boolean {
  return (d.update_seq ?? 0) === 0 && d.timestamp_s >= startedAtS;
}

export function trialPrompt(phase: TrialPhase, truth: { x: number; y: number } | null, trialNo: number): string {
  switch (phase) {
    case "idle": return truth ? `Ready for trial ${trialNo}. Press Start trial.` : "Set the true position first: click the map or type x and y.";
    case "chirping": return "Playing the calibration chirp…";
    case "listening": return truth ? `Make the sound at ${truth.x.toFixed(2)}, ${truth.y.toFixed(2)} now.` : "Make the sound now.";
    case "review": return "Heard it. Check the position, then log the trial.";
  }
}
