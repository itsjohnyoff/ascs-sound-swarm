import type { CalibrationResult, EvalSummary, Room, SessionState, Trial } from "./types";

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = JSON.stringify((await res.json()).detail ?? detail); } catch { /* not json */ }
    throw new Error(`${method} ${path}: ${res.status} ${detail}`);
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
  trials: () => call<{ trials: Trial[]; summary: EvalSummary }>("GET", "/api/eval/trials"),
  addTrial: (t: { label: string; gt_x_m: number; gt_y_m: number; expected_class: string; event_id: string | null }) =>
    call<Trial>("POST", "/api/eval/trials", t),
  // Contract v2 §6: stored separately from trials. Server endpoint may not exist yet.
  addKnownPoint: (t: { label: string; gt_x_m: number; gt_y_m: number; expected_class: string; event_id: string | null }) =>
    call<Trial>("POST", "/api/eval/known-point", t),
  clearTrials: () => call<void>("DELETE", "/api/eval/trials"),
  joinInfo: () => call<{ join_url: string; qr_svg: string }>("GET", "/api/join-info"),
  recordStart: (name: string) => call<{ status: string; path: string }>("POST", "/api/record/start", { name }),
  recordStop: () => call<{ status: string; path: string }>("POST", "/api/record/stop"),
};

export function wsUrl(path: string): string {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;
}
