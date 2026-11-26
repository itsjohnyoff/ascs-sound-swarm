import { api } from "../shared/api";
import { generateChirp, playBuffer } from "../shared/chirpSynth";
import type { LocateResult } from "../shared/types";
import { audioOut, exclusive } from "./calibration";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const TAIL_MS = 900; // let the last chirp and its echoes reach every phone before solving

/**
 * Phones and laptop take turns chirping; the server fits where everyone is.
 * apply=false only checks whether any phone moved since it was last located.
 */
export function runLocate(apply = true): Promise<LocateResult | null> {
  return exclusive(async () => {
    const out = await audioOut(); // fail before arming if the browser blocks sound
    const armed = await api.locateArm(apply);
    const t0 = performance.now();
    const chirp = generateChirp(out.sampleRate);
    for (const slot of armed.laptop_slots) {
      await sleep(Math.max(0, t0 + slot * armed.slot_s * 1000 - performance.now()));
      void playBuffer(out, chirp);
    }
    await sleep(Math.max(0, t0 + armed.duration_s * 1000 + TAIL_MS - performance.now()));
    return api.locateSolve(armed.locate_id);
  });
}

/** One line for the user, naming the phones that need a look. */
export function locateSummary(r: LocateResult, name: (id: string) => string = (id) => id): string {
  if (!r.success) return r.error ?? "Couldn't place the phones. Turn the volume up and try again.";
  const entries = Object.entries(r.phones);
  const failed = entries.filter(([, p]) => p.status === "failed").map(([id]) => name(id));
  const check = entries.filter(([, p]) => p.status === "check").map(([id]) => name(id));
  const moved = entries.filter(([, p]) => p.moved).map(([id, p]) => `${name(id)} (${(p.moved_m ?? 0).toFixed(2)} m)`);
  const parts: string[] = [];
  if (!r.applied && !moved.length) parts.push("No phone has moved.");
  if (moved.length) parts.push(`Moved: ${moved.join(", ")}. Put ${moved.length === 1 ? "it" : "them"} back or locate again.`);
  if (r.applied) parts.push(`Placed ${entries.length - failed.length} of ${entries.length} phones.`);
  if (!r.anchored) parts.push("The phones didn't hear the laptop, so the layout is centred in the room.");
  if (check.length) parts.push(`Check ${check.join(", ")}: the fit is loose.`);
  if (failed.length) parts.push(`Couldn't place ${failed.join(", ")}: drag ${failed.length === 1 ? "it" : "them"} by hand.`);
  if (r.applied) parts.push("If the map is turned or mirrored, use Turn or Mirror.");
  return parts.join(" ");
}
