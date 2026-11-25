import type { PhoneStatus } from "../shared/types";

export interface Say { title: string; detail: string; tone: "ok" | "warn" | "bad" | "idle" }

export function syncCopy(s: Pick<PhoneStatus, "state" | "residual_ms">): Say {
  if (s.state === "synced") {
    return { tone: "ok", title: "Synced", detail: s.residual_ms != null ? `Timing error ${s.residual_ms.toFixed(2)} ms. Keep this screen on.` : "Keep this screen on." };
  }
  if (s.state === "drifting") return { tone: "warn", title: "Re-sync soon", detail: "Timing is drifting. The laptop will chirp again shortly." };
  if (s.state === "clock_jump") return { tone: "bad", title: "Needs re-sync", detail: "This phone lost its timing. Wait for the next chirp." };
  return { tone: "idle", title: "Waiting for sync", detail: "The laptop plays a chirp to sync the phones. Keep the room quiet." };
}

export function micError(err: unknown, secure: boolean): string {
  if (!secure) return "This page needs HTTPS to use the microphone. Open the link from the QR code on the laptop.";
  const name = (err as { name?: string })?.name;
  if (name === "NotAllowedError" || name === "SecurityError") return "Microphone is blocked. Allow microphone access for this site in your browser settings, then reload.";
  if (name === "NotFoundError") return "No microphone found on this device.";
  if (name === "NotReadableError") return "Another app is using the microphone. Close it and try again.";
  return `Couldn't start the microphone. ${(err as Error)?.message ?? ""}`.trim();
}
