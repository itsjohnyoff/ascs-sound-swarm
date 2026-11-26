import { describe, expect, it } from "vitest";
import { centred, distances, speakerOffCentre } from "../src/dashboard/geometry";
import type { Phone, Room } from "../src/shared/types";

const room: Room = { width_m: 8, height_m: 6, temperature_c: 20, speaker_x_m: 4, speaker_y_m: 0.5 };
const phone = (phone_id: string, x_m: number, y_m: number) => ({ phone_id, label: "", x_m, y_m }) as Phone;

describe("speaker centring", () => {
  it("moves the speaker to the middle", () => {
    expect(speakerOffCentre(room)).toBe(true);
    expect(centred(room)).toMatchObject({ speaker_x_m: 4, speaker_y_m: 3 });
    expect(speakerOffCentre(centred(room))).toBe(false);
  });
  it("stays stable for odd sizes so it never loops", () => {
    expect(speakerOffCentre(centred({ ...room, width_m: 7.33, height_m: 5.55 }))).toBe(false);
  });
});

describe("distances", () => {
  it("lists speaker distances then each phone pair once", () => {
    const d = distances([phone("a", 0, 3), phone("b", 4, 0), phone("c", 8, 3)], centred(room));
    expect(d.slice(0, 3).map((x) => x.meters)).toEqual([4, 3, 4]);
    expect(d).toHaveLength(6);
    expect(d[3]).toEqual({ a: "a", b: "b", meters: 5 });
  });
  it("is empty without phones", () => expect(distances([], room)).toEqual([]));
});
