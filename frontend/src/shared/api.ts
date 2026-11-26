import type { CalibrationResult, EvalSummary, LocateArmed, LocateResult, Room, SessionState, Trial } from "./types";

export const OFFLINE_MESSAGE = "Can't reach the server. Check that it's still running on the laptop.";

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error(OFFLINE_MESSAGE);
  }
  if (!res.ok) {
    let detail = "";
    try { const d = (await res.json()).detail; detail = typeof d === "string" ? d : JSON.stringify(d); } catch { /* not json */ }
    throw new Error(detail ? `${detail}` : `The server refused the request (${res.status}).`);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export type Mode = "lfm_dual" | "lfm_single";

export const api = {
  session: () => call<SessionState>("GET", "/api/session"),
  putRoom: (r: Room) => call<Room>("PUT", "/api/room", r),
  putPhone: (id: string, p: { label?: string; x_m: number; y_m: number }) =>
    call("PUT", `/api/phones/${encodeURIComponent(id)}`, p),
  deletePhone: (id: string) => call("DELETE", `/api/phones/${encodeURIComponent(id)}`),
  arm: (mode: Mode) =>
    call<{ calibration_id: string; armed_at_s: number; expected_duration_s: number }>(
      "POST", "/api/calibrate/arm", { mode }),
  solve: (calibration_id: string, mode: Mode, window_s: number) =>
    call<CalibrationResult>("POST", "/api/calibrate/solve", { calibration_id, mode, window_s }),
  locateArm: (apply = true) => call<LocateArmed>("POST", "/api/locate/arm", { apply }),
  locateSolve: (locate_id: string) => call<LocateResult>("POST", "/api/locate/solve", { locate_id }),
  turnLayout: (degrees: number, mirror = false) => call<SessionState>("POST", "/api/layout/turn", { degrees, mirror }),
  trials: () => call<{ trials: Trial[]; summary: EvalSummary }>("GET", "/api/eval/trials"),
  addTrial: (t: { label: string; gt_x_m: number; gt_y_m: number; expected_class: string; event_id: string | null }) =>
    call<Trial>("POST", "/api/eval/trials", t),
  // Contract v2 §6: stored separately from trials. Server endpoint may not exist yet.
  addKnownPoint: (t: { label: string; gt_x_m: number; gt_y_m: number; expected_class: string; event_id: string | null }) =>
    call<Trial>("POST", "/api/eval/known-point", t),
  clearTrials: () => call<void>("DELETE", "/api/eval/trials"),
  joinInfo: () => call<{ join_url: string; qr_svg: string }>("GET", "/api/join-info"),
  recordStart: (name: string) => call<{ status: string; path: string }>("POST", "/api/record/start", { name }),
  recordStatus: () =>
    call<{ status: "idle" | "recording" | "saving" | "awaiting_save"; path: string | null }>("GET", "/api/record/status"),
  recordStop: () => call<{ status: string; path: string }>("POST", "/api/record/stop"),
};

export function wsUrl(path: string): string {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;
}
