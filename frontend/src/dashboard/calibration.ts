import { api, type Mode } from "../shared/api";
import { generateChirp, generateDualChirp, playBuffer } from "../shared/chirpSynth";
import type { CalibrationResult } from "../shared/types";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let ctx: AudioContext | null = null;
let busy = false;

export const isBusy = () => busy;

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
    ctx ??= new AudioContext();
    const fs = ctx.sampleRate;
    await playBuffer(ctx, mode === "lfm_dual" ? generateDualChirp(fs) : generateChirp(fs));
    await sleep(600);
    return await api.solve(armed.calibration_id, mode, mode === "lfm_dual" ? 7 : 4);
  } finally {
    busy = false;
  }
}

export function renderCalTable(table: HTMLElement, r: CalibrationResult) {
  const rows = Object.entries(r.per_phone).map(([id, p]) => {
    const g = grade(p.residual_ms, p.synced);
    const f = (v: number | null, d: number) => (v == null ? "–" : v.toFixed(d));
    return `<tr><td>${id}</td><td>${f(p.residual_ms, 2)} ms</td><td>${f(p.drift_ppm, 1)} ppm</td><td>${f(p.chirp_snr_db, 1)} dB</td><td class="${g}">${g}${p.error ? ` (${p.error})` : ""}</td></tr>`;
  });
  table.innerHTML = `<tr><th>Phone</th><th>Resid.</th><th>Drift</th><th>SNR</th><th></th></tr>${rows.join("")}`;
}
