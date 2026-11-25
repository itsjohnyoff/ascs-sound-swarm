import { FLAG_CLIPPING, FLAG_CLOCK_JUMP, encodeFrame } from "../shared/binaryFrame";
import { wsUrl } from "../shared/api";
import { isDebug } from "../shared/debug";
import type { PhoneStatus, SessionState } from "../shared/types";
import { Diagnostics } from "./diagnostics";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ID_KEY = "ascs_phone_id";

export function getPhoneId(): string {
  try {
    const saved = localStorage.getItem(ID_KEY);
    if (saved) return saved;
  } catch { /* storage blocked */ }
  const id = `phone-${Math.random().toString(36).slice(2, 6)}`;
  try { localStorage.setItem(ID_KEY, id); } catch { /* ignore */ }
  return id;
}

const phoneId = getPhoneId();
const diag = new Diagnostics();
let diagEl: HTMLElement | null = null;
if (isDebug()) {
  diagEl = document.createElement("pre");
  diagEl.id = "diag";
  diagEl.style.cssText = "font:12px ui-monospace,monospace;color:#8b949e;background:#161b22;padding:8px;border-radius:8px;margin:0;text-align:left";
  document.body.appendChild(diagEl);
  setInterval(() => { diagEl!.textContent = diag.lines().join("\n"); }, 500);
}
$("phoneId").textContent = phoneId;
const warnings: string[] = [];
const warn = (m: string) => { warnings.push(m); const w = $("warn"); w.hidden = false; w.textContent = warnings.join(" · "); };
const log = (m: string) => { $("log").textContent = m; };

if (!window.isSecureContext) warn("NOT HTTPS: microphone will be blocked");

let ctx: AudioContext | null = null;
let ws: WebSocket | null = null;
let node: AudioWorkletNode | null = null;
let seq = 0;
let pendingJump = false;
let joined = false;
let retry = 0;
let myPos: { x: number; y: number } | null = null;
let sampleRate = 0;

async function openMic(): Promise<MediaStream> {
  const chain: MediaTrackConstraints[] = [
    { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    { echoCancellation: { ideal: false }, noiseSuppression: { ideal: false }, autoGainControl: { ideal: false } },
    {},
  ];
  let lastErr: unknown;
  for (let i = 0; i < chain.length; i++) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: chain[i]! });
      const s = stream.getAudioTracks()[0]?.getSettings();
      diag.track = { aec: s?.echoCancellation, ns: s?.noiseSuppression, agc: s?.autoGainControl, channels: s?.channelCount };
      if (i > 0) warn(`mic constraints downgraded (level ${i})`);
      if (s && (s.echoCancellation || s.noiseSuppression || s.autoGainControl)) {
        warn(`browser processing ON (AEC=${s.echoCancellation} NS=${s.noiseSuppression} AGC=${s.autoGainControl})`);
      }
      return stream;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function serverPosition(): Promise<{ x: number; y: number; label?: string } | null> {
  try {
    const s = (await (await fetch("/api/session")).json()) as SessionState;
    const p = s.phones.find((q) => q.phone_id === phoneId);
    return p ? { x: p.x_m, y: p.y_m, label: p.label } : null;
  } catch { return null; }
}

async function showPosition() {
  const p = await serverPosition();
  if (p) { myPos = { x: p.x, y: p.y }; }
  $("info").textContent = `${sampleRate} Hz` + (myPos ? ` · server position (${myPos.x.toFixed(2)}, ${myPos.y.toFixed(2)}) m` : "");
}

async function hello() {
  // Server owns positions: reuse the one it already has so reconnects never move us.
  const known = await serverPosition();
  const pos = known ?? { x: 0.5, y: 0.5 };
  ws?.send(JSON.stringify({
    type: "phone_hello", phone_id: phoneId, label: known?.label ?? "",
    x_m: pos.x, y_m: pos.y, sample_rate: sampleRate, device_info: navigator.userAgent.slice(0, 120),
  }));
}

function setSync(s: PhoneStatus) {
  const el = $("sync");
  el.className = "pill";
  if (s.state === "synced" && s.residual_ms != null) { el.textContent = `Synced ${s.residual_ms.toFixed(2)} ms`; el.classList.add("ok"); }
  else if (s.state === "synced") { el.textContent = "Synced"; el.classList.add("ok"); }
  else if (s.state === "drifting") { el.textContent = "Drifting"; el.classList.add("warn"); }
  else { el.textContent = "Needs re-sync"; el.classList.add("bad"); }
}

function connect() {
  ws = new WebSocket(wsUrl(`/ws/phone/${encodeURIComponent(phoneId)}`));
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { retry = 0; void hello().then(showPosition); log("connected"); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") return;
    try {
      const m = JSON.parse(ev.data);
      if (m.type === "phone_status") { setSync(m as PhoneStatus); void showPosition(); }
    } catch { /* ignore */ }
  };
  ws.onclose = () => {
    if (!joined) return;
    $("sync").className = "pill bad"; $("sync").textContent = "Reconnecting…";
    diag.reconnects++;
    const delay = Math.min(8000, 500 * 2 ** retry++);
    pendingJump = true; // gap in the timeline while offline
    setTimeout(connect, delay);
  };
}

async function join() {
  $<HTMLButtonElement>("join").disabled = true;
  try {
    const stream = await openMic();
    ctx = new AudioContext(); // never force a rate: report the real one
    sampleRate = ctx.sampleRate;
    diag.sampleRate = sampleRate;
    await ctx.audioWorklet.addModule("/pcm-worklet.js");
    const src = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, "pcm-worklet", { numberOfInputs: 1, numberOfOutputs: 0 });
    src.connect(node);
    if (ctx.state !== "running") await ctx.resume();
    ctx.onstatechange = () => { if (ctx?.state === "running") pendingJump = true; };
    document.addEventListener("visibilitychange", () => { pendingJump = true; if (!document.hidden) void wake(); });

    node.port.onmessage = (ev) => {
      const { pcm, startSampleIndex, rms, clipping } = ev.data as { pcm: Int16Array; startSampleIndex: number; rms: number; clipping: boolean };
      const bar = $("vuBar");
      bar.style.width = `${Math.min(100, Math.sqrt(rms) * 140)}%`;
      bar.classList.toggle("clip", clipping);
      diag.onBatch(startSampleIndex, pcm.length);
      if (ws?.readyState !== WebSocket.OPEN) { diag.framesDropped++; return; }
      let flags = 0;
      if (clipping) flags |= FLAG_CLIPPING;
      if (pendingJump) { flags |= FLAG_CLOCK_JUMP; pendingJump = false; diag.clockJumps++; }
      ws.send(encodeFrame({
        flags, numSamples: pcm.length, seq: seq++, sampleRate,
        sampleIndex: startSampleIndex, clientTimeMs: performance.timeOrigin + performance.now(),
      }, pcm));
      diag.framesSent++;
    };

    joined = true;
    connect();
    await wake();
    $("join").hidden = true;
    $("sync").textContent = "Joined, waiting for sync";
    $("info").textContent = `${sampleRate} Hz`;
  } catch (e) {
    $<HTMLButtonElement>("join").disabled = false;
    log(`Could not start: ${(e as Error).message}`);
  }
}

async function wake() {
  try { await (navigator as Navigator & { wakeLock?: { request(t: "screen"): Promise<unknown> } }).wakeLock?.request("screen"); }
  catch { /* optional */ }
}

$("join").addEventListener("click", () => void join());
