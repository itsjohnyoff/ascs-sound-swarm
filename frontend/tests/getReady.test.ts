import { describe, expect, it } from "vitest";
import {
  freshSteps, judgeCheck, judgeLevel, judgePhones, judgeSync, readyButtonLabel, runGetReady, type ReadyDeps,
} from "../src/dashboard/getReady";
import type { CalibrationResult, LevelResult, LocateResult, Phone, SessionState } from "../src/shared/types";

const phone = (id: string, o: Partial<Phone> = {}): Phone => ({
  phone_id: id, label: "", x_m: 1, y_m: 1, sample_rate: 48000, connected: true, rms: 0,
  sync_state: "synced", residual_ms: 0.2, drift_ppm: 0, chirp_snr_db: 30, last_sync_ts: 1, ...o,
});
const session = (n: number, gdop: string | null = null): SessionState => ({
  type: "session_state", room: { width_m: 8, height_m: 6, temperature_c: 20, speaker_x_m: 4, speaker_y_m: 3 },
  phones: Array.from({ length: n }, (_, i) => phone(`p${i + 1}`)), gdop_warning: gdop, calibration_status: "synced",
});
const level = (verdicts: Record<string, string>): LevelResult => ({
  ok: Object.values(verdicts).filter((v) => v === "ok").length, total: Object.keys(verdicts).length,
  per_phone: Object.fromEntries(Object.entries(verdicts).map(([id, v]) => [id, { verdict: v as never, snr_db: 30, peak: 0.3 }])),
});
const located: LocateResult = {
  success: true, applied: true, anchored: true, rms_m: 0.01, error: null,
  phones: { p1: { status: "ok", x_m: 1, y_m: 1, rms_m: 0.01, moved_m: null }, p2: { status: "ok", x_m: 7, y_m: 1, rms_m: 0.01, moved_m: null }, p3: { status: "ok", x_m: 4, y_m: 5, rms_m: 0.01, moved_m: null } },
};
const synced = (max = 0.3, ok = true): CalibrationResult => ({
  mode: "lfm_dual", success: ok, synced_count: 3, median_residual_ms: 0.2, max_residual_ms: max,
  per_phone: { p1: { synced: ok, residual_ms: max, drift_ppm: 0, chirp_snr_db: 30, error: null } },
});

function deps(over: Partial<ReadyDeps> = {}): ReadyDeps & { calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    session: () => session(3),
    name: (id) => id.toUpperCase(),
    level: async () => { calls.push("level"); return level({ p1: "ok", p2: "ok", p3: "ok" }); },
    locate: async () => { calls.push("locate"); return located; },
    calibrate: async () => { calls.push("sync"); return synced(); },
    clapCheck: async (id) => { calls.push(`clap:${id}`); return 0.12; },
    ...over,
  };
}

describe("get ready", () => {
  it("runs every step in order and ends ready", async () => {
    const d = deps();
    const updates: string[] = [];
    const steps = await runGetReady(d, freshSteps(), (s) => updates.push(s.map((x) => x.state[0]).join("")));
    expect(steps.every((s) => s.state === "done")).toBe(true);
    expect(d.calls).toEqual(["level", "locate", "sync", "clap:p1"]);
    expect(updates[0]).toBe("rtttt");
    expect(steps.at(-1)!.detail).toBe("The clap landed 0.12 m from P1.");
    expect(readyButtonLabel(steps)).toBe("Run again");
  });

  it("stops at the first problem and Continue resumes from it", async () => {
    let loud = true;
    const d = deps({ level: async () => level(loud ? { p1: "ok", p2: "clipping", p3: "quiet" } : { p1: "ok", p2: "ok", p3: "ok" }) });
    let steps = await runGetReady(d, freshSteps(), () => {});
    expect(steps.map((s) => s.state)).toEqual(["done", "failed", "todo", "todo", "todo"]);
    expect(steps[1]!.detail).toContain("P3 can barely hear the laptop");
    expect(steps[1]!.detail).toContain("P2 is overloaded");
    expect(readyButtonLabel(steps)).toBe("Continue");

    loud = false;
    steps = await runGetReady(d, steps, () => {});
    expect(steps.every((s) => s.state === "done")).toBe(true);
  });

  it("turns a thrown error into a failed step with a way forward", async () => {
    const steps = await runGetReady(deps({ calibrate: async () => { throw new Error("Browser blocked sound."); } }), freshSteps(), () => {});
    expect(steps[3]).toMatchObject({ state: "failed", detail: "Browser blocked sound. Press Continue to try again." });
  });

  it("judges each step with one plain fix", () => {
    expect(judgePhones(session(2))).toEqual({ ok: false, detail: "2 phones connected. Scan the Phone QR on 1 more phone, then press Continue." });
    expect(judgeLevel(level({ a: "silent" })).detail).toContain("turn the laptop volume up");
    expect(judgeSync(synced(2.4)).detail).toBe("Synced, but loose (2.4 ms). Fine to continue.");
    expect(judgeSync(synced(0.3, false)).ok).toBe(false);
    expect(judgeCheck(null, "P1").detail).toContain("Didn't hear the clap");
    expect(judgeCheck(0.8, "P1")).toMatchObject({ ok: false });
  });

  it("collinear phones fail the position step even if the fit worked", async () => {
    const steps = await runGetReady(deps({ session: () => session(3, "collinear") }), freshSteps(), () => {});
    expect(steps[2]).toMatchObject({ state: "failed" });
    expect(steps[2]!.detail).toContain("almost in a line");
  });
});
