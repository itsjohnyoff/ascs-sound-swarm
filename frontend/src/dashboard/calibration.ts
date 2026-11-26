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

/** Run one sound job at a time (calibration, locating phones). Overlapping calls return null. */
export async function exclusive<T>(fn: () => Promise<T>): Promise<T | null> {
  if (busy) return null;
  busy = true;
  try { return await fn(); } finally { busy = false; }
}

/** The running output context, or an error the user can act on. */
export async function audioOut(): Promise<AudioContext> {
  unlockAudio();
  if (ctx!.state !== "running") await ctx!.resume();
  if (ctx!.state !== "running") throw new Error("Browser blocked sound. Click the page once and try again.");
  return ctx!;
}

/** arm, 250 ms lead-in, play, 600 ms tail, solve. Serialized: overlapping runs return null. */
export function runCalibration(mode: Mode): Promise<CalibrationResult | null> {
  return exclusive(async () => {
    const armed = await api.arm(mode);
    await sleep(250);
    const out = await audioOut();
    const fs = out.sampleRate;
    await playBuffer(out, mode === "lfm_dual" ? generateDualChirp(fs) : generateChirp(fs));
    await sleep(600);
    return await api.solve(armed.calibration_id, mode, mode === "lfm_dual" ? 7 : 4);
  });
}

const LABEL = { PASS: "Pass", WARN: "Loose", FAIL: "Fail" } as const;

function failReason(error: string | null): string {
  if (!error) return "";
  return /not detected/i.test(error) ? "Didn't hear the chirp" : error;
}

export function renderCalTable(table: HTMLTableElement, r: CalibrationResult) {
  table.replaceChildren();
  const headings = table.createTHead().insertRow();
  for (const title of ["Phone", "Timing", "Chirp", "Result"]) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = title;
    headings.append(th);
  }
  const body = table.createTBody();
  for (const [id, p] of Object.entries(r.per_phone)) {
    const g = grade(p.residual_ms, p.synced);
    const f = (v: number | null, d: number) => (v == null ? "–" : v.toFixed(d));
    const why = failReason(p.error);
    const row = body.insertRow();
    row.insertCell().textContent = id;
    row.insertCell().textContent = `${f(p.residual_ms, 2)} ms`;
    row.insertCell().textContent = `${f(p.chirp_snr_db, 0)} dB`;
    const status = row.insertCell();
    status.className = g.toLowerCase();
    status.textContent = LABEL[g];
    if (why) {
      const reason = document.createElement("small");
      reason.textContent = why;
      status.append(document.createElement("br"), reason);
    }
  }
}
