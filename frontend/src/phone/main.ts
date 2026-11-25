import { FLAG_CLIPPING, FLAG_CLOCK_JUMP, encodeFrame } from "../shared/binaryFrame";
import { wsUrl } from "../shared/api";
import { isDebug } from "../shared/debug";
import type { PhoneStatus, SessionState } from "../shared/types";
import { Diagnostics } from "./diagnostics";
import { micError, syncCopy, type Say } from "./copy";
import "./style.css";

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
const notice = (m: string, bad = false) => {
  const d = document.createElement("div"); d.className = bad ? "notice bad" : "notice"; d.textContent = m; $("notices").appendChild(d);
};
const warn = (m: string) => notice(m);
const show = (say: Say) => { $("state").dataset.tone = say.tone; $("title").textContent = say.title; $("detail").textContent = say.detail; };
const log = (m: string) => { show({ tone: "bad", title: "Something went wrong", detail: m }); };

if (!window.isSecureContext) notice(micError(null, false), true);

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
      if (i > 0) warn("This phone ignored the clean-audio settings, so timing may be less accurate.");
      if (s && (s.echoCancellation || s.noiseSuppression || s.autoGainControl)) {
        warn("The browser is still filtering the microphone (echo cancel, noise or gain). Claps may be dulled. Try Chrome or Safari with a fresh tab.");
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
  $("info").textContent = myPos ? `Position on the laptop: ${myPos.x.toFixed(2)} m, ${myPos.y.toFixed(2)} m. Sample rate ${sampleRate} Hz.` : `Sample rate ${sampleRate} Hz.`;
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

function setSync(s: PhoneStatus) { show(syncCopy(s)); }

function connect() {
  ws = new WebSocket(wsUrl(`/ws/phone/${encodeURIComponent(phoneId)}`));
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { retry = 0; void hello().then(showPosition); show({ tone: "idle", title: "Waiting for sync", detail: "Connected. The laptop plays a chirp to sync the phones. Keep the room quiet." }); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") return;
    try {
      const m = JSON.parse(ev.data);
      if (m.type === "phone_status") { setSync(m as PhoneStatus); void showPosition(); }
    } catch { /* ignore */ }
  };
  ws.onclose = () => {
    if (!joined) return;
    show({ tone: "warn", title: "Connection lost", detail: "Reconnecting to the laptop. Stay on this page." });
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
    document.addEventListener("visibilitychange", () => {
      pendingJump = true;
      if (!document.hidden) { void wake(); show({ tone: "warn", title: "Needs re-sync", detail: "The screen was off, so timing was lost. Wait for the next chirp." }); }
    });

    node.port.onmessage = (ev) => {
      const { pcm, startSampleIndex, rms, clipping } = ev.data as { pcm: Int16Array; startSampleIndex: number; rms: number; clipping: boolean };
      const bar = $("vuBar");
      bar.style.width = `${Math.min(100, Math.sqrt(rms) * 140)}%`;
      bar.classList.toggle("clip", clipping);
      $("clipNote").textContent = clipping ? "Too loud" : "";
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
    $("vuWrap").hidden = false;
    show({ tone: "idle", title: "Joined", detail: "Microphone is on. Waiting for the laptop to sync the phones." });
  } catch (e) {
    $<HTMLButtonElement>("join").disabled = false;
    show({ tone: "bad", title: "Couldn't start", detail: micError(e, window.isSecureContext) });
  }
}

let wakeWarned = false;
async function wake() {
  const lock = (navigator as Navigator & { wakeLock?: { request(t: "screen"): Promise<unknown> } }).wakeLock;
  try { if (!lock) throw new Error("unsupported"); await lock.request("screen"); }
  catch {
    if (!wakeWarned) { wakeWarned = true; warn("This phone may lock its screen and stop sending audio. Turn off auto-lock in the phone's settings while testing."); }
  }
}

$("join").addEventListener("click", () => void join());
