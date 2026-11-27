import type { Phone } from "../shared/types";

export type Tone = "ok" | "warn" | "bad";
export interface HealthRow { id: string; name: string; delay: string; dropped: string; battery: string; synced: string; tone: Tone }

const SLOW_RTT_MS = 150;
const BAD_RTT_MS = 500;
const LOW_BATTERY = 0.2;
const STALE_SYNC_S = 120;

const isSynced = (p: Phone) => p.sync_state === "synced" || p.sync_state === "drifting";

function ago(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))} s ago`;
  return `${Math.round(seconds / 60)} min ago`;
}

/** One row per connected phone, worst problem decides the tone. */
export function healthRows(phones: Phone[], nowS: number, name: (p: Phone) => string = (p) => p.label || p.phone_id): HealthRow[] {
  return phones.filter((p) => p.connected).map((p) => {
    let tone: Tone = "ok";
    const worse = (t: Tone) => { if (t === "bad" || (t === "warn" && tone === "ok")) tone = t; };
    const rtt = p.rtt_ms ?? null;
    if (rtt != null && rtt >= BAD_RTT_MS) worse("bad"); else if (rtt != null && rtt >= SLOW_RTT_MS) worse("warn");
    const dropped = p.dropped_frames ?? 0;
    if (dropped > 0) worse("warn");
    const battery = p.battery ?? null;
    if (battery != null && battery < LOW_BATTERY && !p.charging) worse("warn");
    if (!isSynced(p)) worse("bad");
    else if (p.last_sync_ts != null && nowS - p.last_sync_ts > STALE_SYNC_S) worse("warn");
    return {
      id: p.phone_id,
      name: name(p),
      delay: rtt == null ? "–" : `${rtt} ms`,
      dropped: String(dropped),
      battery: battery == null ? "–" : `${Math.round(battery * 100)}%${p.charging ? " +" : ""}`,
      synced: !isSynced(p) ? "No" : p.last_sync_ts == null ? "Yes" : ago(nowS - p.last_sync_ts),
      tone,
    };
  });
}

/**
 * Phones that came back (reconnected or lost their clock) while the rest are still synced:
 * a single chirp brings them back without anyone clicking Calibrate.
 */
export function phonesToResync(previous: Phone[], next: Phone[]): string[] {
  const before = new Map(previous.map((p) => [p.phone_id, p]));
  const live = next.filter((p) => p.connected);
  if (!live.some(isSynced)) return []; // nothing synced yet: that's Get ready's job, not a rejoin
  return live
    .filter((p) => !isSynced(p))
    .filter((p) => {
      const was = before.get(p.phone_id);
      return !was || !was.connected || isSynced(was) || p.sync_state === "clock_jump";
    })
    .map((p) => p.phone_id);
}
