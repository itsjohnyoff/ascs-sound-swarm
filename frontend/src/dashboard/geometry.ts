import type { Phone, Room } from "../shared/types";

const round2 = (v: number) => Math.round(v * 100) / 100;

/** The speaker (laptop) always sits in the middle of the room. */
export function roomCentre(room: Pick<Room, "width_m" | "height_m">): { x: number; y: number } {
  return { x: round2(room.width_m / 2), y: round2(room.height_m / 2) };
}

export function centred(room: Room): Room {
  const c = roomCentre(room);
  return { ...room, speaker_x_m: c.x, speaker_y_m: c.y };
}

export function speakerOffCentre(room: Room): boolean {
  const c = roomCentre(room);
  return Math.abs(room.speaker_x_m - c.x) > 0.005 || Math.abs(room.speaker_y_m - c.y) > 0.005;
}

export interface Distance { a: string; b: string; meters: number }

/** Distances worked out from the positions on the map: every phone to the speaker, then every phone pair. */
export function distances(phones: Phone[], room: Room): Distance[] {
  const name = (p: Phone) => p.label || p.phone_id;
  const toSpeaker = phones.map((p) => ({ a: name(p), b: "Speaker", meters: Math.hypot(p.x_m - room.speaker_x_m, p.y_m - room.speaker_y_m) }));
  const pairs = phones.flatMap((p, i) => phones.slice(i + 1).map((q) => ({ a: name(p), b: name(q), meters: Math.hypot(p.x_m - q.x_m, p.y_m - q.y_m) })));
  return [...toSpeaker, ...pairs];
}
