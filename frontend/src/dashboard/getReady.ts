import type { CalibrationResult, LevelResult, LocateResult, SessionState } from "../shared/types";
import { locateSummary } from "./locate";
import { gradeResidual } from "./readiness";

export type StepId = "phones" | "level" | "locate" | "sync" | "check";
export type StepState = "todo" | "running" | "done" | "failed";
export interface Step { id: StepId; label: string; state: StepState; detail: string }

export const MIN_PHONES = 3;
export const CHECK_PASS_M = 0.5;

const LABELS: Record<StepId, string> = {
  phones: "Phones joined",
  level: "Loudness",
  locate: "Phone positions",
  sync: "Clock sync",
  check: "Clap check",
};
export const ORDER: StepId[] = ["phones", "level", "locate", "sync", "check"];

export const freshSteps = (): Step[] => ORDER.map((id) => ({ id, label: LABELS[id], state: "todo", detail: "" }));

type Verdict = { ok: boolean; detail: string };
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;
const live = (s: SessionState | null) => s?.phones.filter((p) => p.connected) ?? [];

export function judgePhones(s: SessionState | null): Verdict {
  const n = live(s).length;
  if (n >= MIN_PHONES) return { ok: true, detail: `${plural(n, "phone")} connected.` };
  return { ok: false, detail: `${plural(n, "phone")} connected. Scan the Phone QR on ${plural(MIN_PHONES - n, "more phone")}, then press Continue.` };
}

export function judgeLevel(r: LevelResult, name: (id: string) => string = (id) => id): Verdict {
  const by = (v: string) => Object.entries(r.per_phone).filter(([, p]) => p.verdict === v).map(([id]) => name(id));
  const quiet = [...by("quiet"), ...by("silent")], loud = by("clipping");
  if (!quiet.length && !loud.length) return { ok: true, detail: `Every phone heard the laptop clearly.` };
  const parts: string[] = [];
  if (quiet.length) parts.push(`${quiet.join(", ")} can barely hear the laptop: turn the laptop volume up.`);
  if (loud.length) parts.push(`${loud.join(", ")} ${loud.length === 1 ? "is" : "are"} overloaded: turn the laptop volume down a little.`);
  return { ok: false, detail: `${parts.join(" ")} Then press Continue.` };
}

export function judgeLocate(r: LocateResult | null, s: SessionState | null, name?: (id: string) => string): Verdict {
  if (!r) return { ok: false, detail: "Another chirp was playing. Press Continue." };
  if (!r.success) return { ok: false, detail: `${locateSummary(r, name)} Then press Continue.` };
  if (s?.gdop_warning) return { ok: false, detail: "The phones are almost in a line. Move them towards the room corners, then press Continue." };
  return { ok: true, detail: locateSummary(r, name) };
}

export function judgeSync(r: CalibrationResult | null): Verdict {
  if (!r) return { ok: false, detail: "Another chirp was playing. Press Continue." };
  const reports = Object.values(r.per_phone);
  const failed = reports.filter((p) => gradeResidual(p.residual_ms, p.synced) === "fail").length;
  if (!r.success || failed) {
    return { ok: false, detail: `${plural(failed || reports.length, "phone")} didn't sync. Turn the laptop volume up, keep the room quiet, then press Continue.` };
  }
  const worst = r.max_residual_ms ?? 0;
  return { ok: true, detail: worst >= 1 ? `Synced, but loose (${worst.toFixed(1)} ms). Fine to continue.` : `Synced to ${worst.toFixed(2)} ms.` };
}

export function judgeCheck(errorM: number | null, phone: string): Verdict {
  if (errorM == null) return { ok: false, detail: `Didn't hear the clap. Clap once, loudly, right next to ${phone}, then press Continue.` };
  if (errorM < CHECK_PASS_M) return { ok: true, detail: `The clap landed ${errorM.toFixed(2)} m from ${phone}.` };
  return { ok: false, detail: `The clap landed ${errorM.toFixed(2)} m from ${phone}. Locate the phones again, then press Continue.` };
}

/** The side effects, so the flow can run against fakes in tests. */
export interface ReadyDeps {
  session(): SessionState | null;
  name(id: string): string;
  level(): Promise<LevelResult>;
  locate(): Promise<LocateResult | null>;
  calibrate(): Promise<CalibrationResult | null>;
  /** Ask the user to clap next to this phone; resolve with the error in metres, or null if nothing was heard. */
  clapCheck(phoneId: string): Promise<number | null>;
}

/** Run from `from` (the first unfinished step) and stop at the first problem. */
export async function runGetReady(deps: ReadyDeps, steps: Step[], onUpdate: (s: Step[]) => void): Promise<Step[]> {
  const start = Math.max(0, steps.findIndex((s) => s.state !== "done"));
  let current = steps.map((s, i) => (i >= start ? { ...s, state: "todo" as StepState, detail: "" } : s));
  const set = (id: StepId, state: StepState, detail: string) => {
    current = current.map((s) => (s.id === id ? { ...s, state, detail } : s));
    onUpdate(current);
  };
  for (const id of ORDER.slice(start)) {
    set(id, "running", RUNNING[id]);
    let v: Verdict;
    try {
      v = await STEP[id](deps);
    } catch (e) {
      v = { ok: false, detail: `${(e as Error).message} Press Continue to try again.` };
    }
    set(id, v.ok ? "done" : "failed", v.detail);
    if (!v.ok) break;
  }
  return current;
}

const RUNNING: Record<StepId, string> = {
  phones: "Counting phones…",
  level: "The laptop plays one chirp…",
  locate: "Phones take turns chirping. Don't touch them…",
  sync: "Syncing clocks…",
  check: "",
};

const STEP: Record<StepId, (d: ReadyDeps) => Promise<Verdict>> = {
  phones: async (d) => judgePhones(d.session()),
  level: async (d) => judgeLevel(await d.level(), d.name),
  locate: async (d) => judgeLocate(await d.locate(), d.session(), d.name),
  sync: async (d) => judgeSync(await d.calibrate()),
  check: async (d) => {
    const target = live(d.session())[0];
    if (!target) return judgePhones(d.session());
    return judgeCheck(await d.clapCheck(target.phone_id), d.name(target.phone_id));
  },
};

export const readyButtonLabel = (steps: Step[]): string =>
  steps.every((s) => s.state === "done") ? "Run again" : steps.some((s) => s.state !== "todo") ? "Continue" : "Get ready";
