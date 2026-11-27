import { describe, expect, it } from "vitest";
import { healthRows, phonesToResync } from "../src/dashboard/health";
import type { Phone } from "../src/shared/types";

const phone = (id: string, o: Partial<Phone> = {}): Phone => ({
  phone_id: id, label: "", x_m: 1, y_m: 1, sample_rate: 48000, connected: true, rms: 0,
  sync_state: "synced", residual_ms: 0.2, drift_ppm: 0, chirp_snr_db: 30, last_sync_ts: 990,
  dropped_frames: 0, rtt_ms: 40, battery: 0.8, charging: false, ...o,
});

describe("phone health", () => {
  it("shows delay, drops, battery and sync age, healthy phones are ok", () => {
    expect(healthRows([phone("a")], 1000)).toEqual([
      { id: "a", name: "a", delay: "40 ms", dropped: "0", battery: "80%", synced: "10 s ago", tone: "ok" },
    ]);
  });

  it("the worst problem sets the tone", () => {
    const rows = healthRows([
      phone("slow", { rtt_ms: 200 }),
      phone("laggy", { rtt_ms: 800 }),
      phone("low", { battery: 0.1 }),
      phone("charging", { battery: 0.1, charging: true }),
      phone("stale", { last_sync_ts: 700 }),
      phone("lost", { sync_state: "clock_jump" }),
      phone("off", { connected: false }),
    ], 1000);
    expect(Object.fromEntries(rows.map((r) => [r.id, r.tone]))).toEqual({
      slow: "warn", laggy: "bad", low: "warn", charging: "ok", stale: "warn", lost: "bad",
    });
    expect(rows.find((r) => r.id === "charging")!.battery).toBe("10% +");
    expect(rows.find((r) => r.id === "stale")!.synced).toBe("5 min ago");
    expect(rows.find((r) => r.id === "lost")!.synced).toBe("No");
  });

  it("unknown values show a dash instead of guessing", () => {
    const [row] = healthRows([phone("ios", { rtt_ms: null, battery: null, last_sync_ts: null })], 1000);
    expect(row).toMatchObject({ delay: "–", battery: "–", synced: "Yes" });
  });
});

describe("auto re-sync after a rejoin", () => {
  it("re-syncs a phone that came back while the others stayed synced", () => {
    const before = [phone("a"), phone("b"), phone("c", { connected: false, sync_state: "unsynced" })];
    const after = [phone("a"), phone("b"), phone("c", { sync_state: "unsynced" })];
    expect(phonesToResync(before, after)).toEqual(["c"]);
  });

  it("re-syncs a phone whose clock jumped (screen locked) and a brand-new phone", () => {
    expect(phonesToResync([phone("a"), phone("b")], [phone("a"), phone("b", { sync_state: "clock_jump" })])).toEqual(["b"]);
    expect(phonesToResync([phone("a")], [phone("a"), phone("new", { sync_state: "unsynced" })])).toEqual(["new"]);
  });

  it("does nothing before the first sync or when nothing changed", () => {
    const fresh = [phone("a", { sync_state: "unsynced" }), phone("b", { sync_state: "unsynced" })];
    expect(phonesToResync(fresh, fresh)).toEqual([]);
    const steady = [phone("a"), phone("b", { sync_state: "unsynced" })];
    expect(phonesToResync(steady, steady)).toEqual([]);
  });
});
