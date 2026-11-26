import { api, OFFLINE_MESSAGE, wsUrl } from "../shared/api";
import { isDebug } from "../shared/debug";
import type { CalibrationResult, Detection, Phone, SessionState, Trial } from "../shared/types";
import { isBusy, renderCalTable, runCalibration, runLevelCheck, unlockAudio } from "./calibration";
import { FloorMap } from "./floorMap";
import { FpsMeter } from "./fps";
import { freshSteps, readyButtonLabel, runGetReady, type ReadyDeps, type StepState } from "./getReady";
import { locateSummary, runLocate } from "./locate";
import { centred, distances, roomCentre, speakerOffCentre } from "./geometry";
import { errorM, insideHull, summarize } from "./judge";
import { nextAction, readiness } from "./readiness";
import { phonesToSpread } from "./placement";
import { isCandidate, isRoomPoint, trialPrompt, waitForTrialChirp, type TrialPhase } from "./trialFlow";
import "./style.css";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const num = (id: string) => parseFloat($<HTMLInputElement>(id).value);
const LAYOUT_KEY = "ascs_layout_v1";
const esc = (t: string) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

let session: SessionState | null = null;
let online = false;
const detections: Detection[] = [];
let truth: { x: number; y: number } | null = null;
let candidate: Detection | null = null;
let trials: Trial[] = [];
let phase: TrialPhase = "idle";
let judge = false;
let trialSaving = false;
let knownPointAfter: string | undefined;

const map = new FloorMap($<HTMLCanvasElement>("map"));

if (isDebug()) {
  const meter = new FpsMeter();
  const el = document.createElement("div");
  el.id = "fps";
  el.style.cssText = "position:fixed;left:8px;bottom:8px;font:12px ui-monospace,monospace;background:#17252a;color:#fff;padding:4px 8px;border-radius:4px";
  document.body.appendChild(el);
  map.onDraw = (now, cost) => meter.record(now, cost);
  (window as unknown as { __fps: FpsMeter }).__fps = meter;
  setInterval(() => { el.textContent = `draw ${meter.fps.toFixed(1)}/s · ${meter.avgCostMs.toFixed(2)} ms avg · ${meter.maxCostMs.toFixed(2)} ms max`; }, 500);
}

/** Only phones we can hear right now. Offline ones are hidden; the server keeps their position for when they rejoin. */
const livePhones = (): Phone[] => (online ? session?.phones.filter((p) => p.connected) ?? [] : []);
function renderDistances() {
  const rows = session ? distances(livePhones(), session.room) : [];
  $("distances").innerHTML = rows.length
    ? rows.map((d) => `<tr><td>${esc(d.a)} to ${esc(d.b)}</td><td>${d.meters.toFixed(2)} m</td></tr>`).join("")
    : `<tr><td>Shown when phones join.</td></tr>`;
}
const refresh = () => {
  if (!session) return;
  map.model = {
    room: session.room, phones: livePhones(), detections,
    showDistances: $<HTMLInputElement>("dist").checked,
    showHeat: $<HTMLInputElement>("heat").checked, showHyperbolas: $<HTMLInputElement>("hyp").checked,
    groundTruth: truth, linkedEstimate: candidate,
  };
  map.invalidate();
};
const say = (id: string, msg: string, isError = false) => { const el = $(id); el.textContent = msg; el.classList.toggle("err", isError); };
const guard = async (id: string, fn: () => Promise<unknown>) => {
  say(id, "");
  try { await fn(); } catch (e) { say(id, (e as Error).message, true); }
};

// ---- status bar -------------------------------------------------------------------------
function renderStatus() {
  const r = readiness(session, online);
  const bar = $("statusBar");
  bar.dataset.tone = r.tone;
  $("headline").textContent = r.headline;
  $("detail").textContent = r.detail;
  markNext();
}
function markNext() {
  let target: string | null = nextAction(session, online, phase);
  if ((!judge && (target === "startBtn" || target === "logBtn")) ||
      (judge && (target === "qrBtn" || target === "readyBtn"))) target = "judgeBtn";
  const next = target ? document.getElementById(target) : null;
  const hint = $<HTMLButtonElement>("nextHint");
  hint.hidden = !next || next.matches(":disabled");
  const previous = document.querySelector("[data-next]");
  if (previous !== next || hint.hidden) previous?.removeAttribute("data-next");
  if (hint.hidden || !next) return;
  next.setAttribute("data-next", "");
  hint.dataset.target = next.id;
  hint.textContent = `Next: ${next.id === "judgeBtn" && judge ? "return to setup" : NEXT_LABEL[next.id]}`;
}
const NEXT_LABEL: Record<string, string> = { qrBtn: "show phone QR", readyBtn: "get ready", judgeBtn: "open judge mode", startBtn: "start trial", logBtn: "log trial" };
$("nextHint").addEventListener("click", () => {
  const el = document.getElementById($("nextHint").dataset.target ?? "");
  el?.scrollIntoView({ block: "center" });
  el?.focus({ preventScroll: true });
});
function flashStatus() {
  const bar = $("statusBar"); bar.classList.remove("flash"); void bar.offsetWidth; bar.classList.add("flash");
}

// ---- session / sockets ---------------------------------------------------------------------
const spread = new Set<string>();
function applySession(s: SessionState) {
  session = s;
  // Phones that join on top of each other get their own spot so every dot is visible.
  for (const m of phonesToSpread(s.phones.filter((p) => p.connected), s.room, spread)) {
    spread.add(m.id);
    const p = s.phones.find((q) => q.phone_id === m.id);
    void api.putPhone(m.id, { label: p?.label, x_m: m.x, y_m: m.y }).catch(() => {});
  }
  if (document.activeElement?.tagName !== "INPUT") fillForms(s);
  // The speaker is always the middle of the room, including after a size change or a server restart.
  if (speakerOffCentre(s.room)) void guard("layoutMsg", () => api.putRoom(centred(s.room)));
  renderPhones(); renderDistances(); renderCalTableLive();
  renderStatus(); renderTrials(); refresh();
}

function fillForms(s: SessionState) {
  const r = s.room;
  const set = (id: string, v: number) => { $<HTMLInputElement>(id).value = String(v); };
  set("rw", r.width_m); set("rh", r.height_m); set("rt", r.temperature_c);
  const c = roomCentre(r);
  $("speakerNote").textContent = `Speaker is the centre of the room (${c.x.toFixed(2)}, ${c.y.toFixed(2)}). Put the laptop there.`;
}

function stateWords(p: Phone): string {
  return ({ synced: "synced", drifting: "drifting", unsynced: "not synced", clock_jump: "needs re-sync" } as const)[p.sync_state];
}

function renderPhones() {
  const box = $("phones");
  if (box.contains(document.activeElement) && document.activeElement?.tagName === "INPUT") return;
  const phones = livePhones();
  if (!phones.length) { box.innerHTML = `<p class="hint">Phones appear here when they join.</p>`; return; }
  box.innerHTML = phones.map((p) => `<div class="phone-row" data-id="${esc(p.phone_id)}">
    <small>${esc(p.phone_id)} · ${stateWords(p)}</small>
    <div><label>Name</label><input data-k="label" value="${esc(p.label)}" placeholder="e.g. North"></div>
    <div><label>x</label><input data-k="x" type="number" step="0.01" value="${p.x_m}"></div>
    <div><label>y</label><input data-k="y" type="number" step="0.01" value="${p.y_m}"></div></div>`).join("");
}

$("phones").addEventListener("change", (ev) => {
  const row = (ev.target as HTMLElement).closest<HTMLElement>("[data-id]"); if (!row) return;
  const q = (k: string) => row.querySelector<HTMLInputElement>(`[data-k=${k}]`)!.value;
  void guard("layoutMsg", () => api.putPhone(row.dataset.id!, { label: q("label"), x_m: parseFloat(q("x")), y_m: parseFloat(q("y")) }));
});

let retry = 0;
function connect() {
  const socket = new WebSocket(wsUrl("/ws/dashboard"));
  socket.onopen = () => { retry = 0; online = true; renderStatus(); };
  socket.onmessage = (ev) => {
    const m = JSON.parse(ev.data as string);
    if (m.type === "session_state") applySession(m as SessionState);
    else if (m.type === "telemetry" && session) {
      for (const t of m.phones as Phone[]) {
        const p = session.phones.find((q) => q.phone_id === t.phone_id);
        if (p) { p.rms = t.rms; p.connected = t.connected; p.sync_state = t.sync_state; }
      }
      refresh();
    } else if (m.type === "calibration_result") showCalibration(m as CalibrationResult);
    else if (m.type === "detection") onDetection(m as Detection);
  };
  socket.onclose = () => { online = false; renderStatus(); renderPhones(); renderDistances(); renderCalTableLive(); refresh(); setTimeout(connect, Math.min(8000, 500 * 2 ** retry++)); };
}

function showCalibration(r: CalibrationResult) {
  const total = Object.keys(r.per_phone).length;
  say("calMsg", r.success
    ? `${r.synced_count} of ${total} phones synced. Worst timing error ${r.max_residual_ms?.toFixed(2) ?? "–"} ms.`
    : "Calibration failed. Check the speaker volume and try again.", !r.success);
  lastCalibration = r;
  renderCalTableLive();
}
let lastCalibration: CalibrationResult | null = null;
/** The calibration table lists only phones that are still connected. */
function renderCalTableLive() {
  if (!lastCalibration) return;
  const live = new Set(livePhones().map((p) => p.phone_id));
  const per_phone = Object.fromEntries(Object.entries(lastCalibration.per_phone).filter(([id]) => live.has(id)));
  const table = $<HTMLTableElement>("calTable");
  if (Object.keys(per_phone).length) renderCalTable(table, { ...lastCalibration, per_phone });
  else table.innerHTML = "";
}

// ---- detections ---------------------------------------------------------------------------
function onDetection(d: Detection) {
  const i = detections.findIndex((x) => x.event_id === d.event_id);
  const isNew = i < 0;
  if (isNew) detections.push(d); else detections[i] = d;
  if (detections.length > 40) detections.shift();
  if (isNew) flashStatus(); // visual only: a laptop beep would be heard by the phones
  if (isNew && detectionWaiter) detectionWaiter(d);
  if (isCandidate(d, phase, isNew)) { candidate = d; phase = "review"; renderTrialPanel(); }
  else if (phase === "review" && candidate?.event_id === d.event_id) { candidate = d; renderTrialPanel(); }
  renderRecent(); refresh();
}

// ---- calibration ----------------------------------------------------------------------------
async function calibrate(mode: "lfm_dual" | "lfm_single"): Promise<CalibrationResult | null> {
  if (isBusy()) return null;
  $<HTMLButtonElement>("calBtn").disabled = true; say("calMsg", "Playing the chirp…");
  markNext();
  try {
    const r = await runCalibration(mode);
    if (r) showCalibration(r);
    return r;
  } finally {
    $<HTMLButtonElement>("calBtn").disabled = false;
    markNext();
  }
}
const showCalError = (e: Error) => say("calMsg", e.message, true);
$("calBtn").addEventListener("click", () => { unlockAudio(); void calibrate("lfm_dual").catch(showCalError); });
// Any click unlocks sound so the repeat chirp and trial chirps can play later without a gesture.
document.addEventListener("pointerdown", () => unlockAudio());
setInterval(() => { if ($<HTMLInputElement>("auto").checked && phase === "idle" && !isBusy() && session?.phones.some((p) => p.connected)) void calibrate("lfm_single").catch(showCalError); }, 30_000);

// ---- locate phones by sound -----------------------------------------------------------------------
const phoneName = (id: string) => session?.phones.find((p) => p.phone_id === id)?.label || id;
async function locate(apply: boolean, msgId: string, button: HTMLButtonElement) {
  if (isBusy()) { say(msgId, "Wait for the current chirp to finish.", true); return; }
  button.disabled = true;
  say(msgId, apply ? "Phones are chirping in turn…" : "Checking positions…");
  try {
    const r = await runLocate(apply);
    if (r) say(msgId, locateSummary(r, phoneName), !r.success);
  } catch (e) {
    say(msgId, (e as Error).message, true);
  } finally {
    button.disabled = false;
  }
}
$("locateBtn").addEventListener("click", () => { unlockAudio(); void locate(true, "locateMsg", $<HTMLButtonElement>("locateBtn")); });
$("checkMovedBtn").addEventListener("click", () => { unlockAudio(); void locate(false, "movedMsg", $<HTMLButtonElement>("checkMovedBtn")); });
$("turnLeft").addEventListener("click", () => void guard("locateMsg", () => api.turnLayout(-15)));
$("turnRight").addEventListener("click", () => void guard("locateMsg", () => api.turnLayout(15)));
$("mirrorBtn").addEventListener("click", () => void guard("locateMsg", () => api.turnLayout(0, true)));

// ---- get ready -----------------------------------------------------------------------------------
let steps = freshSteps();
let readyRunning = false;
const STATE_WORD: Record<StepState, string> = { todo: "", running: "Working", done: "Done", failed: "Fix" };
function renderSteps() {
  $("steps").innerHTML = steps.map((s) => `<li data-state="${s.state}"><span class="step-name">${esc(s.label)}</span>
    <span class="step-state">${STATE_WORD[s.state]}</span>${s.detail ? `<p>${esc(s.detail)}</p>` : ""}</li>`).join("");
  const button = $<HTMLButtonElement>("readyBtn");
  button.textContent = readyRunning ? "Working…" : readyButtonLabel(steps);
  button.disabled = readyRunning;
}

/** Resolve with the next new detection, or null after timeoutMs. */
let detectionWaiter: ((d: Detection) => void) | null = null;
const nextDetection = (timeoutMs: number) => new Promise<Detection | null>((resolve) => {
  const timer = setTimeout(() => { detectionWaiter = null; resolve(null); }, timeoutMs);
  detectionWaiter = (d) => { clearTimeout(timer); detectionWaiter = null; resolve(d); };
});

const readyDeps: ReadyDeps = {
  session: () => (online ? session : null),
  name: phoneName,
  level: runLevelCheck,
  locate: () => runLocate(true),
  calibrate: () => calibrate("lfm_dual"),
  clapCheck: async (id) => {
    const target = session?.phones.find((p) => p.phone_id === id);
    if (!target) return null;
    steps = steps.map((s) => (s.id === "check" ? { ...s, detail: `Clap once, loudly, right next to ${phoneName(id)}. Its screen says "Clap right next to me".` } : s));
    renderSteps();
    await api.hint(id, { tone: "warn", title: "Clap right next to me", detail: "Once, loudly." }).catch(() => {});
    const d = await nextDetection(20_000);
    if (!d) return null;
    const err = Math.hypot(d.x_m - target.x_m, d.y_m - target.y_m);
    await api.addKnownPoint({ label: "Get ready clap", gt_x_m: target.x_m, gt_y_m: target.y_m, expected_class: "clap", event_id: d.event_id }).catch(() => {});
    await api.hint(id, err < 0.5 ? { tone: "ok", title: "Got it", detail: "Ready. Don't move this phone." }
      : { tone: "bad", title: "That landed too far away", detail: "Follow the laptop." }).catch(() => {});
    return err;
  },
};
$("readyBtn").addEventListener("click", () => {
  if (readyRunning) return;
  unlockAudio();
  readyRunning = true; renderSteps();
  void runGetReady(readyDeps, steps, (s) => { steps = s; renderSteps(); })
    .then((s) => { steps = s; })
    .finally(() => { readyRunning = false; renderSteps(); markNext(); });
});
renderSteps();

// ---- room, layout ----------------------------------------------------------------------------
$("roomBtn").addEventListener("click", () => void guard("layoutMsg", () => api.putRoom(centred({
  width_m: num("rw"), height_m: num("rh"), temperature_c: num("rt"), speaker_x_m: 0, speaker_y_m: 0,
}))));
map.onMovePhone = (id, x, y) => { const p = session?.phones.find((q) => q.phone_id === id); void guard("layoutMsg", () => api.putPhone(id, { label: p?.label, x_m: x, y_m: y })); };
$("heat").addEventListener("change", refresh); $("hyp").addEventListener("change", refresh); $("dist").addEventListener("change", refresh);

$("saveLayout").addEventListener("click", () => {
  if (!session) return;
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify({ room: session.room, phones: session.phones.map((p) => ({ id: p.phone_id, label: p.label, x: p.x_m, y: p.y_m })) }));
    say("layoutMsg", `Saved the room and ${session.phones.length} phone positions.`);
  } catch { say("layoutMsg", "This browser won't let the page save. Write the positions down instead.", true); }
});
$("restoreLayout").addEventListener("click", () => void guard("layoutMsg", async () => {
  let saved: { room: SessionState["room"]; phones: { id: string; label: string; x: number; y: number }[] } | null = null;
  try { saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? "null"); } catch { /* ignore */ }
  if (!saved) throw new Error("Nothing saved yet. Press Save layout once the positions are right.");
  await api.putRoom(centred(saved.room));
  for (const p of saved.phones) await api.putPhone(p.id, { label: p.label, x_m: p.x, y_m: p.y });
  say("layoutMsg", `Restored the room and ${saved.phones.length} phone positions. Phones that haven't joined yet keep their place.`);
}));

$("qrBtn").addEventListener("click", () => void guard("layoutMsg", async () => {
  const j = await api.joinInfo();
  $("qrBody").innerHTML = j.qr_svg;
  const link = $<HTMLAnchorElement>("qrUrl");
  link.href = j.join_url;
  link.textContent = j.join_url;
  $<HTMLDialogElement>("qr").showModal();
}));

// ---- recorder ----------------------------------------------------------------------------------
const recordName = $<HTMLInputElement>("recName");
const recordStart = $<HTMLButtonElement>("recStart");
const recordStop = $<HTMLButtonElement>("recStop");
async function restoreRecording(announce: boolean) {
  const current = await api.recordStatus();
  recordName.disabled = recordStart.disabled = current.status !== "idle";
  recordStop.disabled = current.status === "idle" || current.status === "saving";
  recordStop.textContent = current.status === "awaiting_save" ? "Retry save" : "Stop and save";
  if (announce && current.path) {
    const action = current.status === "recording" ? "Recording to"
      : current.status === "saving" ? "Saving" : "Ready to retry saving";
    say("recMsg", `${action} ${current.path}.`);
  } else if (announce) {
    say("recMsg", "");
  }
  if (current.status === "saving") {
    setTimeout(() => { void restoreRecording(announce).catch((e: Error) => say("recMsg", e.message, true)); }, 500);
  }
}
recordStart.addEventListener("click", () => void guard("recMsg", async () => {
  recordStart.disabled = true;
  const name = recordName.value.trim() || "spike-01";
  try {
    const r = await api.recordStart(name);
    say("recMsg", `Recording to ${r.path}. Play your chirps and claps, then stop.`);
    recordName.disabled = true;
    recordStop.disabled = false;
    recordStop.textContent = "Stop and save";
  } catch (e) {
    recordStart.disabled = false;
    await restoreRecording(false).catch(() => {});
    throw e;
  }
}));
recordStop.addEventListener("click", () => void guard("recMsg", async () => {
  recordStop.disabled = true;
  say("recMsg", "Saving the recording…");
  try {
    const r = await api.recordStop();
    say("recMsg", `Saved ${r.path}. Check it with: python -m backend.dsp.replay ${r.path}`);
    recordName.disabled = false;
    recordStart.disabled = false;
  } catch (e) {
    recordStop.disabled = false;
    await restoreRecording(false).catch(() => {});
    throw e;
  }
}));
void restoreRecording(true).catch(() => { recordStart.disabled = false; });

// ---- judge ------------------------------------------------------------------------------------------
$("judgeBtn").addEventListener("click", () => {
  judge = !judge; document.body.classList.toggle("judge", judge); map.judgeMode = judge;
  $("judgeBtn").textContent = judge ? "Leave judge mode" : "Judge mode";
  $("judgeBtn").setAttribute("aria-pressed", String(judge));
  map.invalidate(); markNext();
});
map.onPick = (x, y) => setTruth(Math.round(x * 100) / 100, Math.round(y * 100) / 100);
function setTruth(x: number, y: number, updateFields = true) {
  if (trialSaving) return;
  truth = { x, y };
  knownPointAfter = detections.at(-1)?.event_id;
  $("knownBadge").textContent = "";
  if (updateFields) {
    $<HTMLInputElement>("gx").value = String(x);
    $<HTMLInputElement>("gy").value = String(y);
  }
  const live = session?.phones.filter((p) => p.connected) ?? [];
  say("hullNote", live.length >= 3 && !insideHull(x, y, live) ? "This point is outside the area the phones surround. Expect a larger error." : "");
  renderTrialPanel(); refresh();
}
const typed = () => {
  const x = num("gx"), y = num("gy");
  if (session && isRoomPoint(session.room, x, y)) {
    setTruth(x, y, false);
    return;
  }
  truth = null;
  $("knownBadge").textContent = "";
  say("hullNote", Number.isFinite(x) && Number.isFinite(y) ? "Enter coordinates inside the room." : "");
  renderTrialPanel();
  refresh();
};
$("gx").addEventListener("input", typed); $("gy").addEventListener("input", typed);

const fmtTime = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const chip = (c: string) => `<span class="chip ${c}">${c === "drone" ? "Drone" : c === "clap" ? "Clap" : "Other"}</span>`;

function renderTrialPanel() {
  $("prompt").textContent = trialPrompt(phase, truth, trials.length + 1);
  const btns = $("trialBtns");
  const b = (id: string, label: string, cls = "") =>
    `<button id="${id}" class="${cls}" ${trialSaving || (!truth && id !== "cancelBtn") ? "disabled" : ""}>${label}</button>`;
  if (phase === "idle") btns.innerHTML = b("startBtn", "Start trial");
  else if (phase === "chirping") btns.innerHTML = "";
  else if (phase === "listening") btns.innerHTML = b("missBtn", "Mark as missed", "quiet") + b("cancelBtn", "Cancel", "quiet");
  else {
    const e = candidate && truth ? errorM(candidate, truth.x, truth.y) : null;
    $("prompt").innerHTML = candidate ? `${chip(candidate.sound_class)} at ${candidate.x_m.toFixed(2)}, ${candidate.y_m.toFixed(2)}<div class="result">${e != null ? `${e.toFixed(2)} m off` : ""}</div>` : "";
    btns.innerHTML = b("logBtn", "Log trial") + b("otherBtn", "Pick another", "quiet") + b("missBtn", "Mark as missed", "quiet");
  }
  if (phase !== "review") $("recent").innerHTML = "";
  markNext();
}

$("trialBtns").addEventListener("click", (ev) => {
  const id = (ev.target as HTMLElement).closest("button")?.id;
  if (id === "startBtn") void startTrial();
  else if (id === "cancelBtn") { phase = "idle"; renderTrialPanel(); }
  else if (id === "missBtn") void logTrial(null);
  else if (id === "logBtn" && candidate) void logTrial(candidate);
  else if (id === "otherBtn") renderRecent(true);
});

async function startTrial() {
  if (!truth) { say("status", "Set the true position first: click the map or type x and y.", true); return; }
  say("status", "");
  if ($<HTMLInputElement>("pre").checked) {
    phase = "chirping"; renderTrialPanel();
    try {
      unlockAudio();
      await waitForTrialChirp(() => calibrate("lfm_single"));
    } catch (e) {
      say("status", (e as Error).message, true);
      phase = "idle"; renderTrialPanel();
      return;
    }
  }
  phase = "listening"; renderTrialPanel();
}

function body(ev: Detection | null) {
  if (!truth) throw new Error("Set the true position first.");
  return { label: `Position ${trials.length + 1}`, gt_x_m: truth.x, gt_y_m: truth.y, expected_class: $<HTMLSelectElement>("gc").value, event_id: ev?.event_id ?? null };
}

async function logTrial(ev: Detection | null) {
  if (trialSaving || !truth) return;
  trialSaving = true;
  $<HTMLInputElement>("gx").disabled = true;
  $<HTMLInputElement>("gy").disabled = true;
  $<HTMLSelectElement>("gc").disabled = true;
  say("status", "Saving trial…");
  renderTrialPanel();
  try {
    const saved = await api.addTrial(body(ev));
    trials = [...trials.filter((trial) => trial.trial_idx !== saved.trial_idx), saved];
    say("status", saved.error_m == null ? `Position ${saved.trial_idx} logged as missed.`
      : `Position ${saved.trial_idx} logged: ${saved.error_m.toFixed(2)} m off.`);
    truth = null; candidate = null; phase = "idle";
    $<HTMLInputElement>("gx").value = ""; $<HTMLInputElement>("gy").value = ""; say("hullNote", "");
    renderTrials();
    refresh();
  } catch (e) {
    say("status", (e as Error).message, true);
  } finally {
    trialSaving = false;
    $<HTMLInputElement>("gx").disabled = false;
    $<HTMLInputElement>("gy").disabled = false;
    $<HTMLSelectElement>("gc").disabled = false;
    renderTrialPanel();
  }
}

function renderRecent(show = false) {
  if (!show && phase !== "review") { $("recent").innerHTML = ""; return; }
  if (!show) return;
  const last5 = detections.slice(-5).reverse();
  $("recent").innerHTML = `<p class="hint">Last ${last5.length} detections</p><table>` + last5.map((d) =>
    `<tr><td>${fmtTime(d.timestamp_s)}</td><td>${chip(d.sound_class)}</td><td>${d.x_m.toFixed(2)}, ${d.y_m.toFixed(2)}</td><td><button class="quiet" data-ev="${esc(d.event_id)}">Use</button></td></tr>`).join("") + "</table>";
}
$("recent").addEventListener("click", (ev) => {
  const id = (ev.target as HTMLElement).dataset.ev; if (!id) return;
  const d = detections.find((x) => x.event_id === id); if (!d) return;
  candidate = d; $("recent").innerHTML = ""; renderTrialPanel(); refresh();
});

$("knownBtn").addEventListener("click", () => void guard("status", async () => {
  const point = truth;
  const d = candidate ?? detections.at(-1);
  if (!point || !d || (!candidate && d.event_id === knownPointAfter)) {
    throw new Error("Set the true position, then make a sound so there is a detection to check.");
  }
  const button = $<HTMLButtonElement>("knownBtn");
  button.disabled = true;
  $("knownBadge").textContent = "";
  try {
    const saved = await api.addKnownPoint({ ...body(d), label: "Known point" });
    const err = saved.error_m;
    if (truth !== point || err == null) return;
    $("knownBadge").innerHTML = `<span class="badge ${err < 0.5 ? "pass" : "fail"}">${err.toFixed(2)} m · ${err < 0.5 ? "within 0.5 m" : "too far, recheck positions"}</span>`;
  } finally {
    button.disabled = false;
  }
}));

async function loadTrials() { const r = await api.trials(); trials = r.trials; renderTrials(); }
function renderTrials() {
  const s = summarize(trials); const f = (v: number | null) => (v == null ? "–" : v.toFixed(2));
  $("cards").innerHTML = [[`${trials.length}/10`, "logged"], [f(s.mean), "mean m"], [f(s.p90), "p90 m"], [String(s.missed), "missed"]]
    .map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join("");
  $("strip").innerHTML = Array.from({ length: 10 }, (_, i) => {
    const t = trials[i];
    const cls = !t ? "" : t.missed ? "miss" : (t.error_m ?? 9) < 0.5 ? "pass" : (t.error_m ?? 9) < 1 ? "warn" : "fail";
    return `<span class="${cls}"></span>`;
  }).join("");
  const hull = session?.phones.filter((p) => p.connected) ?? [];
  $("trials").innerHTML = trials.length ? `<tr><th>#</th><th>True x, y</th><th>Error</th><th>Heard</th><th>Inside</th></tr>` + trials.map((t) =>
    `<tr><td>${t.trial_idx}</td><td>${t.gt_x_m.toFixed(1)}, ${t.gt_y_m.toFixed(1)}</td><td>${t.missed ? "Missed" : f(t.error_m) + " m"}</td><td>${t.pred_class ? chip(t.pred_class) : "–"}</td><td>${(t.inside_hull ?? insideHull(t.gt_x_m, t.gt_y_m, hull)) ? "Yes" : "No"}</td></tr>`).join("") : "";
}
$("resetBtn").addEventListener("click", () => {
  if (trialSaving || !trials.length || !confirm(`Clear all ${trials.length} logged trials? This can't be undone.`)) return;
  void guard("status", async () => { await api.clearTrials(); await loadTrials(); renderTrialPanel(); });
});

// ---- boot ----------------------------------------------------------------------------------------
void api.session().then((s) => { online = true; applySession(s); }).catch(() => { say("status", OFFLINE_MESSAGE, true); renderStatus(); });
void loadTrials().catch(() => {});
void fetch("/api/detections/recent?limit=5").then((r) => r.json()).then((j: { detections: Detection[] }) => {
  for (const d of [...j.detections].reverse()) if (!detections.some((x) => x.event_id === d.event_id)) detections.push(d);
  refresh();
}).catch(() => {});
renderStatus(); renderTrialPanel(); renderTrials();
connect();
