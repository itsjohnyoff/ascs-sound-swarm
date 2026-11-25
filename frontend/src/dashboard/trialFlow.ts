import type { CalibrationResult, Detection, Room } from "../shared/types";

export type TrialPhase = "idle" | "chirping" | "listening" | "review";

export function isRoomPoint(room: Room, x: number, y: number): boolean {
  return x >= 0 && x <= room.width_m && y >= 0 && y <= room.height_m;
}

/** Evaluate a live WebSocket message against the current phase without comparing wall clocks. */
export function isCandidate(d: Detection, phase: TrialPhase, isNew: boolean): boolean {
  return phase === "listening" && isNew && (d.update_seq ?? 0) === 0;
}

export async function waitForTrialChirp(calibrate: () => Promise<Pick<CalibrationResult, "success"> | null>): Promise<void> {
  const result = await calibrate();
  if (!result) throw new Error("A calibration is already running. Wait and try again.");
  if (!result.success) throw new Error("Calibration failed. Check the speaker volume and try again.");
  // The server suppresses onsets for CHIRP_SUPPRESS_S after the last sample used by the solve.
  await new Promise((resolve) => setTimeout(resolve, 300));
}

export function trialPrompt(phase: TrialPhase, truth: { x: number; y: number } | null, trialNo: number): string {
  switch (phase) {
    case "idle": return truth ? `Ready for trial ${trialNo}. Press Start trial.` : "Set the true position first: click the map or type x and y.";
    case "chirping": return "Playing the calibration chirp…";
    case "listening": return truth ? `Make the sound at ${truth.x.toFixed(2)}, ${truth.y.toFixed(2)} now.` : "Make the sound now.";
    case "review": return "Heard it. Check the position, then log the trial.";
  }
}
