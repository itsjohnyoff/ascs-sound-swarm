import { api, wsUrl } from "../shared/api";
import type { CalibrationResult, Detection, Phone, SessionState, Trial } from "../shared/types";
import { isBusy, renderCalTable, runCalibration, unlockAudio } from "./calibration";
import { isDebug } from "../shared/debug";
import { FloorMap } from "./floorMap";
import { FpsMeter } from "./fps";
import { errorM, insideHull, summarize } from "./judge";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const num = (id: string) => parseFloat($<HTMLInputElement>(id).value);

let session: SessionState | null = null;
const detections: Detection[] = [];
let truth: { x: number; y: number } | null = null;
let linked: Detection | null = null;
let trials: Trial[] = [];
let judge = false;

const map = new FloorMap($<HTMLCanvasElement>("map"));
if (isDebug()) {
  const meter = new FpsMeter();
  const el = document.createElement("div");
  el.id = "fps";
  el.style.cssText = "position:fixed;left:8px;bottom:8px;font:12px ui-monospace,monospace;background:#161b22;color:#8b949e;padding:4px 8px;border-radius:6px";
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
    groundTruth: truth, linkedEstimate: linked,
  };
  map.invalidate();
};
const say = (id: string, msg: string) => { $(id).textContent = msg; };
const guard = async (id: string, fn: () => Promise<unknown>) => {
  try { await fn(); } catch (e) { say(id, (e as Error).message); }
};

// ---- session / sockets -------------------------------------------------------------------
function applySession(s: SessionState, fromEditing = false) {
  session = s;
  const g = $("gdop"); g.style.display = s.gdop_warning ? "block" : "none"; g.textContent = `Geometry warning: ${s.gdop_warning ?? ""}`;
  say("status", `${s.phones.filter((p) => p.connected).length} phones · ${s.calibration_status}`);
  if (!fromEditing && document.activeElement?.tagName !== "INPUT") fillForms(s);
  renderPhones(s.phones);
  refresh();
}

function fillForms(s: SessionState) {
  const r = s.room;
  const set = (id: string, v: number) => { $<HTMLInputElement>(id).value = String(v); };
  set("rw", r.width_m); set("rh", r.height_m); set("rt", r.temperature_c); set("sx", r.speaker_x_m); set("sy", r.speaker_y_m);
}

function renderPhones(phones: Phone[]) {
  const box = $("phones");
  if (box.contains(document.activeElement) && document.activeElement?.tagName === "INPUT") return; // don't clobber typing
  box.innerHTML = phones.map((p) => `<div class="row" style="margin-bottom:4px" data-id="${p.phone_id}">
    <div style="flex:2"><label>${p.phone_id} · ${p.sync_state}${p.connected ? "" : " · offline"}</label><input data-k="label" value="${p.label}" placeholder="label"></div>
    <div><label>x</label><input data-k="x" type="number" step="0.01" value="${p.x_m}"></div>
    <div><label>y</label><input data-k="y" type="number" step="0.01" value="${p.y_m}"></div></div>`).join("");
}

$("phones").addEventListener("change", (ev) => {
  const row = (ev.target as HTMLElement).closest<HTMLElement>("[data-id]"); if (!row) return;
  const q = (k: string) => row.querySelector<HTMLInputElement>(`[data-k=${k}]`)!.value;
  void guard("status", () => api.putPhone(row.dataset.id!, { label: q("label"), x_m: parseFloat(q("x")), y_m: parseFloat(q("y")) }));
});

let socket: WebSocket | null = null; let retry = 0;
function connect() {
  socket = new WebSocket(wsUrl("/ws/dashboard"));
  socket.onopen = () => { retry = 0; };
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
  socket.onclose = () => setTimeout(connect, Math.min(8000, 500 * 2 ** retry++));
}

function showCalibration(r: CalibrationResult) {
  say("calMsg", r.success ? `${r.synced_count} synced · median ${r.median_residual_ms?.toFixed(2) ?? "–"} ms · max ${r.max_residual_ms?.toFixed(2) ?? "–"} ms` : "Calibration failed");
  renderCalTable($("calTable"), r);
}

function onDetection(d: Detection) {
  const i = detections.findIndex((x) => x.event_id === d.event_id);
  if (i >= 0) detections[i] = d; else detections.push(d); // sustained updates reuse event_id
  if (detections.length > 40) detections.shift();
  renderRecent();
  refresh();
}

// ---- calibration ---------------------------------------------------------------------------
async function calibrate(mode: "lfm_dual" | "lfm_single") {
  if (isBusy()) return;
  $<HTMLButtonElement>("calBtn").disabled = true; say("calMsg", "Chirping…");
  try {
    const r = await runCalibration(mode);
    if (r) showCalibration(r);
  } catch (e) { say("calMsg", (e as Error).message); }
  finally { $<HTMLButtonElement>("calBtn").disabled = false; }
}
$("calBtn").addEventListener("click", () => { unlockAudio(); void calibrate("lfm_dual"); });
// Any click unlocks sound so auto-chirp and judge chirps can play later without a gesture.
document.addEventListener("pointerdown", () => unlockAudio());
setInterval(() => { if ($<HTMLInputElement>("auto").checked && !isBusy() && session?.phones.some((p) => p.connected)) void calibrate("lfm_single"); }, 30_000);

// ---- room panel ----------------------------------------------------------------------------
$("roomBtn").addEventListener("click", () => void guard("status", () => api.putRoom({
  width_m: num("rw"), height_m: num("rh"), temperature_c: num("rt"), speaker_x_m: num("sx"), speaker_y_m: num("sy"),
})));
map.onMovePhone = (id, x, y) => { const p = session?.phones.find((q) => q.phone_id === id); void guard("status", () => api.putPhone(id, { label: p?.label, x_m: x, y_m: y })); };
map.onMoveSpeaker = (x, y) => { if (session) void guard("status", () => api.putRoom({ ...session!.room, speaker_x_m: x, speaker_y_m: y })); };
$("heat").addEventListener("change", refresh); $("hyp").addEventListener("change", refresh);

// ---- QR ------------------------------------------------------------------------------------
$("qrBtn").addEventListener("click", () => void guard("status", async () => {
  const j = await api.joinInfo(); $("qrBody").innerHTML = j.qr_svg; $("qrUrl").textContent = j.join_url;
  $<HTMLDialogElement>("qr").showModal();
}));

// ---- recorder (Gate 0) -----------------------------------------------------------------------
$("recStart").addEventListener("click", () => void guard("recMsg", async () => {
  const r = await api.recordStart($<HTMLInputElement>("recName").value.trim() || "spike-01");
  say("recMsg", `Recording → ${r.path}`); $<HTMLButtonElement>("recStart").disabled = true; $<HTMLButtonElement>("recStop").disabled = false;
}));
$("recStop").addEventListener("click", () => void guard("recMsg", async () => {
  const r = await api.recordStop();
  say("recMsg", `Saved ${r.path}. Run: python -m backend.dsp.replay ${r.path}`);
  $<HTMLButtonElement>("recStart").disabled = false; $<HTMLButtonElement>("recStop").disabled = true;
}));

// ---- judge console -------------------------------------------------------------------------
$("judgeBtn").addEventListener("click", () => {
  judge = !judge; document.body.classList.toggle("judge", judge); map.judgeMode = judge;
  $("judgeBtn").textContent = judge ? "Exit judge mode" : "Judge mode"; map.invalidate();
});
map.onPick = (x, y) => setTruth(Math.round(x * 100) / 100, Math.round(y * 100) / 100);
function setTruth(x: number, y: number) {
  truth = { x, y }; linked = null; $<HTMLInputElement>("gx").value = String(x); $<HTMLInputElement>("gy").value = String(y); refresh();
}
const typed = () => { const x = num("gx"), y = num("gy"); if (isFinite(x) && isFinite(y)) setTruth(x, y); };
$("gx").addEventListener("change", typed); $("gy").addEventListener("change", typed);

const fmtTime = (t: number) => new Date(t * 1000).toLocaleTimeString();
const badge = (c: string) => `<span class="badge ${c.toUpperCase()}">${c.toUpperCase()}</span>`;

function renderRecent() {
  const last5 = detections.slice(-5).reverse();
  $("recent").innerHTML = last5.length
    ? last5.map((d) => `<div class="row" style="margin:3px 0"><span style="flex:3">${fmtTime(d.timestamp_s)} ${badge(d.sound_class)} ${(d.class_confidence * 100).toFixed(0)}% (${d.x_m.toFixed(2)}, ${d.y_m.toFixed(2)})</span><button data-ev="${d.event_id}" style="flex:1">Use</button></div>`).join("")
    : `<div style="color:var(--mute)">No detections yet</div>`;
}
$("recent").addEventListener("click", (ev) => {
  const id = (ev.target as HTMLElement).dataset.ev; if (!id) return;
  const d = detections.find((x) => x.event_id === id); if (!d) return;
  linked = d; refresh(); void logTrial(d);
});
$("missBtn").addEventListener("click", () => void logTrial(null));

function body(ev: Detection | null) {
  if (!truth) throw new Error("Set ground truth first (click map or type x, y)");
  return { label: `Pos ${trials.length + 1}`, gt_x_m: truth.x, gt_y_m: truth.y, expected_class: $<HTMLSelectElement>("gc").value, event_id: ev?.event_id ?? null };
}

async function logTrial(ev: Detection | null) {
  await guard("status", async () => { await api.addTrial(body(ev)); await loadTrials(); });
  if ($<HTMLInputElement>("pre").checked) void calibrate("lfm_single"); // next trial starts with a chirp (CONTRACTS 2)
}

$("knownBtn").addEventListener("click", () => void guard("status", async () => {
  const d = linked ?? detections[detections.length - 1];
  if (!truth || !d) throw new Error("Need ground truth and a detection");
  const err = errorM(d, truth.x, truth.y);
  $("knownBadge").innerHTML = `<span class="badge ${err < 0.5 ? "PASS" : "FAIL"}" style="background:${err < 0.5 ? "#1a7f37" : "#b62324"}">${err.toFixed(2)} m ${err < 0.5 ? "OK" : "CHECK"}</span>`;
  try { await api.addKnownPoint({ ...body(d), label: "known-point" }); } catch { /* server endpoint optional until Agent 1 ships it */ }
}));

async function loadTrials() {
  const r = await api.trials(); trials = r.trials; renderTrials();
}
function renderTrials() {
  const s = summarize(trials); const f = (v: number | null) => (v == null ? "–" : v.toFixed(2));
  $("cards").innerHTML = [[`${trials.length}/10`, "done"], [f(s.mean), "mean m"], [f(s.median), "median m"], [f(s.p90), "p90 m"], [String(s.missed), "missed"]]
    .map(([v, l]) => `<div class="card"><b>${v}</b>${l}</div>`).join("");
  const hull = session?.phones.filter((p) => p.connected) ?? [];
  $("trials").innerHTML = `<tr><th>#</th><th>truth</th><th>err</th><th>class</th><th>hull</th></tr>` + trials.map((t) =>
    `<tr><td>${t.trial_idx}</td><td>${t.gt_x_m.toFixed(1)},${t.gt_y_m.toFixed(1)}</td><td>${t.missed ? "MISSED" : f(t.error_m)}</td><td>${t.pred_class ? badge(t.pred_class) : "–"}</td><td>${(t.inside_hull ?? insideHull(t.gt_x_m, t.gt_y_m, hull)) ? "in" : "out"}</td></tr>`).join("");
}
$("resetBtn").addEventListener("click", () => { if (confirm("Clear all trials?")) void guard("status", async () => { await api.clearTrials(); await loadTrials(); }); });

// ---- boot ----------------------------------------------------------------------------------
void api.session().then((s) => applySession(s)).catch(() => {});
void loadTrials().catch(() => {});
// After a dashboard reload, restore the last detections so the judge list is not empty.
void fetch("/api/detections/recent?limit=5").then((r) => r.json()).then((j: { detections: Detection[] }) => {
  for (const d of [...j.detections].reverse()) if (!detections.some((x) => x.event_id === d.event_id)) detections.push(d);
  renderRecent(); refresh();
}).catch(() => {});
renderRecent();
connect();
