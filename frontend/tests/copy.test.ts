import { describe, expect, it } from "vitest";
import { nextAction, readiness } from "../src/dashboard/readiness";
import { isCandidate, trialPrompt } from "../src/dashboard/trialFlow";
import { micError, syncCopy } from "../src/phone/copy";
import type { Detection, Phone, SessionState } from "../src/shared/types";

const phone = (o: Partial<Phone> = {}): Phone => ({
  phone_id: "p", label: "", x_m: 1, y_m: 1, sample_rate: 48000, connected: true, rms: 0,
  sync_state: "synced", residual_ms: 0.2, drift_ppm: 0, chirp_snr_db: 20, last_sync_ts: 1, ...o,
});
const session = (phones: Phone[], gdop: string | null = null): SessionState => ({
  type: "session_state", room: { width_m: 8, height_m: 6, temperature_c: 20, speaker_x_m: 4, speaker_y_m: 0.5 },
  phones, gdop_warning: gdop, calibration_status: "synced",
});

describe("readiness", () => {
  it("offline wins", () => expect(readiness(session([phone()]), false).tone).toBe("bad"));
  it("no phones", () => expect(readiness(session([]), true).headline).toBe("No phones yet"));
  it("needs three", () => expect(readiness(session([phone(), phone()]), true).detail).toContain("Add 1 more phone"));
  it("unsynced phones prompt calibration", () => {
    const r = readiness(session([phone(), phone(), phone({ sync_state: "unsynced", residual_ms: null })]), true);
    expect(r.headline).toBe("2 of 3 phones synced");
    expect(r.detail).toContain("Calibrate");
  });
  it("ready when all synced under 1 ms", () => {
    const r = readiness(session([phone(), phone(), phone(), phone()]), true);
    expect(r.tone).toBe("ok"); expect(r.headline).toBe("Ready. 4 phones synced");
  });
  it("loose and bad timing", () => {
    expect(readiness(session([phone(), phone(), phone({ residual_ms: 2 })]), true).tone).toBe("warn");
    expect(readiness(session([phone(), phone(), phone({ residual_ms: 7 })]), true).tone).toBe("bad");
  });
  it("geometry warning", () => expect(readiness(session([phone(), phone(), phone()], "collinear"), true).headline).toBe("Move the phones apart"));
});

describe("next action", () => {
  it("guides the operator in order", () => {
    expect(nextAction(session([phone()]), true, "idle")).toBe("qrBtn");
    expect(nextAction(session([phone(), phone(), phone({ sync_state: "unsynced" })]), true, "idle")).toBe("calBtn");
    expect(nextAction(session([phone(), phone(), phone()]), true, "idle")).toBe("startBtn");
    expect(nextAction(session([phone(), phone(), phone()]), true, "review")).toBe("logBtn");
    expect(nextAction(session([phone(), phone(), phone()]), true, "listening")).toBeNull();
    expect(nextAction(null, true, "idle")).toBeNull();
  });
});

describe("trial flow", () => {
  const d = (o: Partial<Detection>) => ({ timestamp_s: 100, update_seq: 0, ...o }) as Detection;
  it("accepts only new first-emission detections", () => {
    expect(isCandidate(d({}), 99)).toBe(true);
    expect(isCandidate(d({}), 101)).toBe(false);
    expect(isCandidate(d({ update_seq: 2 }), 99)).toBe(false);
  });
  it("prompts", () => {
    expect(trialPrompt("idle", null, 1)).toContain("Set the true position");
    expect(trialPrompt("listening", { x: 2, y: 3 }, 1)).toBe("Make the sound at 2.00, 3.00 now.");
  });
});

describe("phone copy", () => {
  it("sync states", () => {
    expect(syncCopy({ state: "synced", residual_ms: 0.16 }).detail).toContain("0.16 ms");
    expect(syncCopy({ state: "clock_jump", residual_ms: null }).title).toBe("Needs re-sync");
    expect(syncCopy({ state: "unsynced", residual_ms: null }).tone).toBe("idle");
  });
  it("mic errors say how to fix", () => {
    expect(micError({ name: "NotAllowedError" }, true)).toContain("Allow microphone access");
    expect(micError({}, false)).toContain("HTTPS");
  });
});
