import type { Phone, Room } from "../shared/types";

const SAME_SPOT_M = 0.05;
const KEEP_APART_M = 0.5;

/** Spots around the room edge, inset from the walls, in the order new phones should take them. */
export function slots(room: Room): { x: number; y: number }[] {
  const { width_m: w, height_m: h } = room;
  const i = Math.min(1, w / 4, h / 4);
  return [
    [i, i], [w - i, i], [w - i, h - i], [i, h - i],
    [w / 2, i], [w / 2, h - i], [i, h / 2], [w - i, h / 2],
  ].map(([x, y]) => ({ x: Math.round(x! * 100) / 100, y: Math.round(y! * 100) / 100 }));
}

/**
 * Phones that joined on top of another phone get a free spot instead, so every dot is visible.
 * Only phones that are connected, not yet synced, and sharing a position are moved; the first one stays put.
 */
export function phonesToSpread(phones: Phone[], room: Room, alreadyPlaced: Set<string>): { id: string; x: number; y: number }[] {
  const taken = phones.map((p) => ({ x: p.x_m, y: p.y_m }));
  const out: { id: string; x: number; y: number }[] = [];
  const free = slots(room);
  phones.forEach((p, idx) => {
    if (!p.connected || p.sync_state !== "unsynced" || alreadyPlaced.has(p.phone_id)) return;
    const stacked = phones.slice(0, idx).some((q) => Math.hypot(q.x_m - p.x_m, q.y_m - p.y_m) < SAME_SPOT_M);
    if (!stacked) return;
    const spot = free.find((s) => taken.every((t) => Math.hypot(t.x - s.x, t.y - s.y) >= KEEP_APART_M));
    if (!spot) return;
    taken.push(spot);
    out.push({ id: p.phone_id, ...spot });
  });
  return out;
}
