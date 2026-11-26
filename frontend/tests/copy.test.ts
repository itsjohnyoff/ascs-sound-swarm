import { describe, expect, it, vi } from "vitest";
import { nextAction, readiness } from "../src/dashboard/readiness";
import { isCandidate, isRoomPoint, trialPrompt, waitForTrialChirp } from "../src/dashboard/trialFlow";
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
    expect(r.detail).toContain("Get ready");
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
    expect(nextAction(session([phone(), phone(), phone({ sync_state: "unsynced" })]), true, "idle")).toBe("readyBtn");
    expect(nextAction(session([phone(), phone(), phone()]), true, "idle")).toBe("startBtn");
    expect(nextAction(session([phone(), phone(), phone()]), true, "review")).toBe("logBtn");
    expect(nextAction(session([phone(), phone(), phone()]), true, "listening")).toBeNull();
    expect(nextAction(null, true, "idle")).toBeNull();
  });
  it("waits on poor geometry and suggests recalibration on loose timing", () => {
    expect(nextAction(session([phone(), phone(), phone()], "collinear"), true, "idle")).toBeNull();
    expect(nextAction(session([phone(), phone(), phone({ residual_ms: 1 })]), true, "idle")).toBe("readyBtn");
  });
});

describe("trial flow", () => {
  const d = (o: Partial<Detection> = {}) => ({ timestamp_s: 100, update_seq: 0, ...o }) as Detection;
  it("accepts only new first-emission detections received while listening, regardless of clock skew", () => {
    expect(isCandidate(d({ timestamp_s: -3600 }), "listening", true)).toBe(true);
    expect(isCandidate(d({ timestamp_s: 3600 }), "chirping", true)).toBe(false);
    expect(isCandidate(d(), "listening", false)).toBe(false);
    expect(isCandidate(d({ update_seq: 2 }), "listening", true)).toBe(false);
  });
  it("requires a successful solve before a chirped trial can start listening", async () => {
    await expect(waitForTrialChirp(async () => ({ success: false }))).rejects.toThrow("Calibration failed");
    await expect(waitForTrialChirp(async () => null)).rejects.toThrow("already running");
    await expect(waitForTrialChirp(async () => { throw new Error("Server unavailable"); })).rejects.toThrow("Server unavailable");
  });
  it("waits out the server's suppression window after a successful solve", async () => {
    vi.useFakeTimers();
    try {
      const ready = vi.fn();
      const listening = waitForTrialChirp(async () => ({ success: true })).then(ready);
      await vi.advanceTimersByTimeAsync(299);
      expect(ready).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1);
      await listening;
      expect(ready).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
  });
  it("prompts", () => {
    expect(trialPrompt("idle", null, 1)).toContain("Set the true position");
    expect(trialPrompt("listening", { x: 2, y: 3 }, 1)).toBe("Make the sound at 2.00, 3.00 now.");
  });
  it("accepts room edges and rejects incomplete or off-map coordinates", () => {
    const room = session([]).room;
    expect(isRoomPoint(room, 0, 6)).toBe(true);
    expect(isRoomPoint(room, 8, 0)).toBe(true);
    expect(isRoomPoint(room, 8.01, 1)).toBe(false);
    expect(isRoomPoint(room, -0.01, 1)).toBe(false);
    expect(isRoomPoint(room, 1, NaN)).toBe(false);
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

import { phonesToSpread, slots } from "../src/dashboard/placement";

describe("placement", () => {
  const room = session([]).room;
  it("moves stacked unsynced phones to free edge spots, keeping the first", () => {
    const ph = [phone({ phone_id: "a", x_m: 0, y_m: 0, sync_state: "unsynced" }), phone({ phone_id: "b", x_m: 0, y_m: 0, sync_state: "unsynced" }), phone({ phone_id: "c", x_m: 0, y_m: 0, sync_state: "unsynced" })];
    const moves = phonesToSpread(ph, room, new Set());
    expect(moves.map((m) => m.id)).toEqual(["b", "c"]);
    expect(moves[0]).toMatchObject(slots(room)[0]!);
    expect(Math.hypot(moves[0]!.x - moves[1]!.x, moves[0]!.y - moves[1]!.y)).toBeGreaterThan(0.5);
  });
  it("leaves synced, offline, already-placed and lone phones alone", () => {
    const stacked = [phone({ phone_id: "a", sync_state: "unsynced" }), phone({ phone_id: "b", sync_state: "unsynced" })];
    expect(phonesToSpread([stacked[0]!], room, new Set())).toEqual([]);
    expect(phonesToSpread([stacked[0]!, { ...stacked[1]!, sync_state: "synced" }], room, new Set())).toEqual([]);
    expect(phonesToSpread([stacked[0]!, { ...stacked[1]!, connected: false }], room, new Set())).toEqual([]);
    expect(phonesToSpread(stacked, room, new Set(["b"]))).toEqual([]);
  });
});
