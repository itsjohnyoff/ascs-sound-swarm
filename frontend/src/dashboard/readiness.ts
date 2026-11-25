import type { SessionState } from "../shared/types";

export type Tone = "ok" | "warn" | "bad" | "idle";
export interface Readiness { tone: Tone; headline: string; detail: string }

const MIN_PHONES = 3;

export function gradeResidual(residualMs: number | null | undefined, synced: boolean): "pass" | "warn" | "fail" {
  if (!synced || residualMs == null) return "fail";
  return residualMs < 1 ? "pass" : residualMs < 5 ? "warn" : "fail";
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** One sentence that says whether it is safe to start judging. */
export function readiness(s: SessionState | null, online: boolean): Readiness {
  if (!online) return { tone: "bad", headline: "Can't reach the server", detail: "Check that the server is still running on the laptop. This page reconnects by itself." };
  if (!s) return { tone: "idle", headline: "Connecting…", detail: "" };
  const live = s.phones.filter((p) => p.connected);
  if (live.length === 0) return { tone: "idle", headline: "No phones yet", detail: "Open Phone QR and scan it on each phone, then tap Join." };
  if (live.length < MIN_PHONES) {
    const need = MIN_PHONES - live.length;
    return { tone: "warn", headline: `${plural(live.length, "phone")} connected`, detail: `Add ${plural(need, "more phone")} to locate sounds. Three is the minimum, four or more is better.` };
  }
  if (s.gdop_warning) return { tone: "warn", headline: "Move the phones apart", detail: s.gdop_warning };
  const synced = live.filter((p) => p.sync_state === "synced" || p.sync_state === "drifting");
  if (synced.length < live.length) {
    const lost = live.length - synced.length;
    return { tone: "warn", headline: `${synced.length} of ${live.length} phones synced`, detail: `${plural(lost, "phone")} need${lost === 1 ? "s" : ""} a chirp. Press Calibrate and keep the room quiet.` };
  }
  const worst = Math.max(...synced.map((p) => p.residual_ms ?? 0));
  if (worst >= 5) return { tone: "bad", headline: `Timing is off by ${worst.toFixed(1)} ms`, detail: "Calibrate again. If it stays high, move the laptop speaker closer to the phones." };
  if (worst >= 1) return { tone: "warn", headline: `Ready, but timing is loose (${worst.toFixed(1)} ms)`, detail: "Calibrate again for a tighter result before judging." };
  return { tone: "ok", headline: `Ready. ${plural(live.length, "phone")} synced`, detail: `Worst timing error ${worst.toFixed(2)} ms.` };
}

export type NextAction = "qrBtn" | "calBtn" | "startBtn" | "logBtn" | null;

/** Which button the operator should press next; the dashboard frames it. */
export function nextAction(s: SessionState | null, online: boolean, phase: "idle" | "chirping" | "listening" | "review"): NextAction {
  if (!online || !s) return null;
  if (phase === "review") return "logBtn";
  if (phase !== "idle") return null;
  const live = s.phones.filter((p) => p.connected);
  if (live.length < MIN_PHONES) return "qrBtn";
  const synced = live.filter((p) => p.sync_state === "synced" || p.sync_state === "drifting");
  if (synced.length < live.length) return "calBtn";
  return "startBtn";
}
