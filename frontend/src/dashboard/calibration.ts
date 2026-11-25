import { api, type Mode } from "../shared/api";
import { generateChirp, generateDualChirp, playBuffer } from "../shared/chirpSynth";
import type { CalibrationResult } from "../shared/types";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let ctx: AudioContext | null = null;
let busy = false;

export const isBusy = () => busy;

/** Create/resume the output AudioContext. Must run synchronously inside a user gesture (Safari/iOS). */
export function unlockAudio(): void {
  ctx ??= new AudioContext();
  if (ctx.state !== "running") void ctx.resume();
}

export function grade(residualMs: number | null | undefined, synced: boolean): "PASS" | "WARN" | "FAIL" {
  if (!synced || residualMs == null) return "FAIL";
  return residualMs < 1 ? "PASS" : residualMs < 5 ? "WARN" : "FAIL";
}

/** arm, 250 ms lead-in, play, 600 ms tail, solve. Serialized: overlapping runs return null. */
export async function runCalibration(mode: Mode): Promise<CalibrationResult | null> {
  if (busy) return null;
  busy = true;
  try {
    const armed = await api.arm(mode);
    await sleep(250);
    unlockAudio();
    if (ctx!.state !== "running") await ctx!.resume();
    if (ctx!.state !== "running") throw new Error("Browser blocked sound. Click the page once and try again.");
    const fs = ctx!.sampleRate;
    await playBuffer(ctx!, mode === "lfm_dual" ? generateDualChirp(fs) : generateChirp(fs));
    await sleep(600);
    return await api.solve(armed.calibration_id, mode, mode === "lfm_dual" ? 7 : 4);
  } finally {
    busy = false;
  }
}

const LABEL = { PASS: "Pass", WARN: "Loose", FAIL: "Fail" } as const;

function failReason(error: string | null): string {
  if (!error) return "";
  return /not detected/i.test(error) ? "Didn't hear the chirp" : error;
}

export function renderCalTable(table: HTMLElement, r: CalibrationResult) {
  const rows = Object.entries(r.per_phone).map(([id, p]) => {
    const g = grade(p.residual_ms, p.synced);
    const f = (v: number | null, d: number) => (v == null ? "–" : v.toFixed(d));
    const why = failReason(p.error);
    return `<tr><td>${id}</td><td>${f(p.residual_ms, 2)} ms</td><td>${f(p.chirp_snr_db, 0)} dB</td><td class="${g.toLowerCase()}">${LABEL[g]}${why ? `<br><small>${why}</small>` : ""}</td></tr>`;
  });
  table.innerHTML = `<tr><th>Phone</th><th>Timing</th><th>Chirp</th><th>Result</th></tr>${rows.join("")}`;
}
