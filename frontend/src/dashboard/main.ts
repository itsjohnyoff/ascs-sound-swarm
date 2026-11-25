import { api, OFFLINE_MESSAGE, wsUrl } from "../shared/api";
import { isDebug } from "../shared/debug";
import type { CalibrationResult, Detection, Phone, SessionState, Trial } from "../shared/types";
import { isBusy, renderCalTable, runCalibration, unlockAudio } from "./calibration";
import { FloorMap } from "./floorMap";
import { FpsMeter } from "./fps";
import { errorM, insideHull, summarize } from "./judge";
import { readiness } from "./readiness";
import { isCandidate, trialPrompt, type TrialPhase } from "./trialFlow";
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
let trialStartedS = 0;
let judge = false;

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

const refresh = () => {
  if (!session) return;
  map.model = {
    room: session.room, phones: session.phones, detections,
    showHeat: $<HTMLInputElement>("heat").checked, showHyperbolas: $<HTMLInputElement>("hyp").checked,
    groundTruth: truth, linkedEstimate: candidate,
  };
  map.invalidate();
};
const say = (id: string, msg: string, isError = false) => { const el = $(id); el.textContent = msg; el.classList.toggle("err", isError); };
const guard = async (id: string, fn: () => Promise<unknown>) => {
  try { await fn(); say(id, ""); } catch (e) { say(id, (e as Error).message, true); }
};

// ---- status bar -------------------------------------------------------------------------
function renderStatus() {
  const r = readiness(session, online);
  const bar = $("statusBar");
  bar.dataset.tone = r.tone;
  $("headline").textContent = r.headline;
  $("detail").textContent = r.detail;
}
function flashStatus() {
  const bar = $("statusBar"); bar.classList.remove("flash"); void bar.offsetWidth; bar.classList.add("flash");
}

// ---- session / sockets ---------------------------------------------------------------------
function applySession(s: SessionState) {
  session = s;
  if (document.activeElement?.tagName !== "INPUT") fillForms(s);
  renderPhones(s.phones);
  renderStatus(); renderTrials(); refresh();
}

function fillForms(s: SessionState) {
  const r = s.room;
  const set = (id: string, v: number) => { $<HTMLInputElement>(id).value = String(v); };
  set("rw", r.width_m); set("rh", r.height_m); set("rt", r.temperature_c); set("sx", r.speaker_x_m); set("sy", r.speaker_y_m);
}

function stateWords(p: Phone): string {
  if (!p.connected) return "offline";
  return ({ synced: "synced", drifting: "drifting", unsynced: "not synced", clock_jump: "needs re-sync" } as const)[p.sync_state];
}

function renderPhones(phones: Phone[]) {
  const box = $("phones");
  if (box.contains(document.activeElement) && document.activeElement?.tagName === "INPUT") return;
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
  socket.onclose = () => { online = false; renderStatus(); setTimeout(connect, Math.min(8000, 500 * 2 ** retry++)); };
}

function showCalibration(r: CalibrationResult) {
  const total = Object.keys(r.per_phone).length;
  say("calMsg", r.success
    ? `${r.synced_count} of ${total} phones synced. Worst timing error ${r.max_residual_ms?.toFixed(2) ?? "–"} ms.`
    : "Calibration failed. Check the speaker volume and try again.", !r.success);
  renderCalTable($("calTable"), r);
}

// ---- detections ---------------------------------------------------------------------------
function onDetection(d: Detection) {
  const i = detections.findIndex((x) => x.event_id === d.event_id);
  const isNew = i < 0;
  if (isNew) detections.push(d); else detections[i] = d;
  if (detections.length > 40) detections.shift();
  if (isNew) flashStatus(); // visual only: a laptop beep would be heard by the phones
  if (phase === "listening" && isCandidate(d, trialStartedS)) { candidate = d; phase = "review"; renderTrialPanel(); }
  else if (phase === "review" && candidate?.event_id === d.event_id) { candidate = d; renderTrialPanel(); }
  renderRecent(); refresh();
}

// ---- calibration ----------------------------------------------------------------------------
async function calibrate(mode: "lfm_dual" | "lfm_single") {
  if (isBusy()) return;
  $<HTMLButtonElement>("calBtn").disabled = true; say("calMsg", "Playing the chirp…");
  try {
    const r = await runCalibration(mode);
    if (r) showCalibration(r);
  } catch (e) { say("calMsg", (e as Error).message, true); }
  finally { $<HTMLButtonElement>("calBtn").disabled = false; }
}
$("calBtn").addEventListener("click", () => { unlockAudio(); void calibrate("lfm_dual"); });
// Any click unlocks sound so the repeat chirp and trial chirps can play later without a gesture.
document.addEventListener("pointerdown", () => unlockAudio());
setInterval(() => { if ($<HTMLInputElement>("auto").checked && phase === "idle" && !isBusy() && session?.phones.some((p) => p.connected)) void calibrate("lfm_single"); }, 30_000);

// ---- room, layout ----------------------------------------------------------------------------
$("roomBtn").addEventListener("click", () => void guard("layoutMsg", () => api.putRoom({
  width_m: num("rw"), height_m: num("rh"), temperature_c: num("rt"), speaker_x_m: num("sx"), speaker_y_m: num("sy"),
})));
map.onMovePhone = (id, x, y) => { const p = session?.phones.find((q) => q.phone_id === id); void guard("layoutMsg", () => api.putPhone(id, { label: p?.label, x_m: x, y_m: y })); };
map.onMoveSpeaker = (x, y) => { if (session) void guard("layoutMsg", () => api.putRoom({ ...session!.room, speaker_x_m: x, speaker_y_m: y })); };
$("heat").addEventListener("change", refresh); $("hyp").addEventListener("change", refresh);

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
  await api.putRoom(saved.room);
  for (const p of saved.phones) await api.putPhone(p.id, { label: p.label, x_m: p.x, y_m: p.y });
  say("layoutMsg", `Restored the room and ${saved.phones.length} phone positions. Phones that haven't joined yet keep their place.`);
}));

$("qrBtn").addEventListener("click", () => void guard("layoutMsg", async () => {
  const j = await api.joinInfo(); $("qrBody").innerHTML = j.qr_svg; $("qrUrl").textContent = j.join_url;
  $<HTMLDialogElement>("qr").showModal();
}));

// ---- recorder ----------------------------------------------------------------------------------
$("recStart").addEventListener("click", () => void guard("recMsg", async () => {
  const r = await api.recordStart($<HTMLInputElement>("recName").value.trim() || "spike-01");
  say("recMsg", `Recording to ${r.path}. Play your chirps and claps, then stop.`);
  $<HTMLButtonElement>("recStart").disabled = true; $<HTMLButtonElement>("recStop").disabled = false;
}));
$("recStop").addEventListener("click", () => void guard("recMsg", async () => {
  const r = await api.recordStop();
  say("recMsg", `Saved ${r.path}. Check it with: python -m backend.dsp.replay ${r.path}`);
  $<HTMLButtonElement>("recStart").disabled = false; $<HTMLButtonElement>("recStop").disabled = true;
}));

// ---- judge ------------------------------------------------------------------------------------------
$("judgeBtn").addEventListener("click", () => {
  judge = !judge; document.body.classList.toggle("judge", judge); map.judgeMode = judge;
  $("judgeBtn").textContent = judge ? "Leave judge mode" : "Judge mode"; map.invalidate();
});
map.onPick = (x, y) => setTruth(Math.round(x * 100) / 100, Math.round(y * 100) / 100);
function setTruth(x: number, y: number) {
  truth = { x, y }; candidate = null;
  if (phase === "review") phase = "listening";
  $<HTMLInputElement>("gx").value = String(x); $<HTMLInputElement>("gy").value = String(y);
  const live = session?.phones.filter((p) => p.connected) ?? [];
  say("hullNote", live.length >= 3 && !insideHull(x, y, live) ? "This point is outside the area the phones surround. Expect a larger error." : "");
  renderTrialPanel(); refresh();
}
const typed = () => { const x = num("gx"), y = num("gy"); if (isFinite(x) && isFinite(y)) setTruth(x, y); };
$("gx").addEventListener("change", typed); $("gy").addEventListener("change", typed);

const fmtTime = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const chip = (c: string) => `<span class="chip ${c}">${c === "drone" ? "Drone" : c === "clap" ? "Clap" : "Other"}</span>`;

function renderTrialPanel() {
  $("prompt").textContent = trialPrompt(phase, truth, trials.length + 1);
  const btns = $("trialBtns");
  const b = (id: string, label: string, cls = "") => `<button id="${id}" class="${cls}">${label}</button>`;
  if (phase === "idle") btns.innerHTML = b("startBtn", "Start trial");
  else if (phase === "chirping") btns.innerHTML = "";
  else if (phase === "listening") btns.innerHTML = b("missBtn", "Mark as missed", "quiet") + b("cancelBtn", "Cancel", "quiet");
  else {
    const e = candidate && truth ? errorM(candidate, truth.x, truth.y) : null;
    $("prompt").innerHTML = candidate ? `${chip(candidate.sound_class)} at ${candidate.x_m.toFixed(2)}, ${candidate.y_m.toFixed(2)}<div class="result">${e != null ? `${e.toFixed(2)} m off` : ""}</div>` : "";
    btns.innerHTML = b("logBtn", "Log trial") + b("otherBtn", "Pick another", "quiet") + b("missBtn", "Mark as missed", "quiet");
  }
  if (phase !== "review") $("recent").innerHTML = "";
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
  unlockAudio();
  trialStartedS = Date.now() / 1000;
  if ($<HTMLInputElement>("pre").checked) {
    phase = "chirping"; renderTrialPanel();
    try { await calibrate("lfm_single"); } catch (e) { say("status", (e as Error).message, true); }
    trialStartedS = Date.now() / 1000; // chirp detections are suppressed server-side; count only what follows
  }
  phase = "listening"; renderTrialPanel();
}

function body(ev: Detection | null) {
  if (!truth) throw new Error("Set the true position first.");
  return { label: `Position ${trials.length + 1}`, gt_x_m: truth.x, gt_y_m: truth.y, expected_class: $<HTMLSelectElement>("gc").value, event_id: ev?.event_id ?? null };
}

async function logTrial(ev: Detection | null) {
  try {
    await api.addTrial(body(ev));
    const n = trials.length + 1;
    say("status", ev && truth ? `Position ${n} logged: ${errorM(ev, truth.x, truth.y).toFixed(2)} m off.` : `Position ${n} logged as missed.`);
    truth = null; candidate = null; phase = "idle";
    $<HTMLInputElement>("gx").value = ""; $<HTMLInputElement>("gy").value = ""; say("hullNote", "");
    await loadTrials(); renderTrialPanel(); refresh();
  } catch (e) { say("status", (e as Error).message, true); }
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
  const d = candidate ?? detections[detections.length - 1];
  if (!truth || !d) throw new Error("Set the true position, then make a sound so there is a detection to check.");
  const err = errorM(d, truth.x, truth.y);
  $("knownBadge").innerHTML = `<span class="badge ${err < 0.5 ? "pass" : "fail"}">${err.toFixed(2)} m · ${err < 0.5 ? "within 0.5 m" : "too far, recheck positions"}</span>`;
  try { await api.addKnownPoint({ ...body(d), label: "Known point" }); } catch { /* the badge above already shows the result */ }
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
$("resetBtn").addEventListener("click", () => { if (confirm(`Clear all ${trials.length} logged trials? This can't be undone.`)) void guard("status", async () => { await api.clearTrials(); await loadTrials(); }); });

// ---- boot ----------------------------------------------------------------------------------------
void api.session().then((s) => { online = true; applySession(s); }).catch(() => { say("status", OFFLINE_MESSAGE, true); renderStatus(); });
void loadTrials().catch(() => {});
void fetch("/api/detections/recent?limit=5").then((r) => r.json()).then((j: { detections: Detection[] }) => {
  for (const d of [...j.detections].reverse()) if (!detections.some((x) => x.event_id === d.event_id)) detections.push(d);
  refresh();
}).catch(() => {});
renderStatus(); renderTrialPanel(); renderTrials();
connect();
