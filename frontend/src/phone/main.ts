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
let retryTimer: ReturnType<typeof setTimeout> | undefined;
let micStream: MediaStream | null = null;
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
    } catch (e) {
      if (e instanceof DOMException && e.name === "NotAllowedError") throw e;
      lastErr = e;
    }
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
  const socket = (ws = new WebSocket(wsUrl(`/ws/phone/${encodeURIComponent(phoneId)}`)));
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
    if (!joined || socket !== ws) return; // ignore sockets we already replaced
    show({ tone: "warn", title: "Connection lost", detail: "Reconnecting to the laptop. Stay on this page." });
    diag.reconnects++;
    const delay = Math.min(8000, 500 * 2 ** retry++);
    pendingJump = true; // gap in the timeline while offline
    clearTimeout(retryTimer);
    retryTimer = setTimeout(connect, delay);
  };
}

async function join() {
  const button = $<HTMLButtonElement>("join");
  button.disabled = true;
  button.textContent = "Starting microphone…";
  let stream: MediaStream | undefined;
  try {
    stream = await openMic();
    micStream = stream;
    // The phone can take the microphone away (call, another app, long lock): offer to join again.
    stream.getAudioTracks()[0]?.addEventListener("ended", () => leave("The microphone stopped. Tap Join to connect again."));
    ctx = new AudioContext(); // never force a rate: report the real one
    sampleRate = ctx.sampleRate;
    diag.sampleRate = sampleRate;
    await ctx.audioWorklet.addModule("/pcm-worklet.js");
    const src = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, "pcm-worklet", { numberOfInputs: 1, numberOfOutputs: 0 });
    src.connect(node);
    if (ctx.state !== "running") await ctx.resume();
    ctx.onstatechange = () => { if (ctx?.state === "running") pendingJump = true; };
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

    show({ tone: "idle", title: "Connecting to the laptop", detail: "Microphone is on. Keep this page open." });
    connect();
    joined = true;
    button.hidden = true;
    $("vuWrap").hidden = false;
    void wake();
  } catch (e) {
    stream?.getTracks().forEach((track) => track.stop());
    node?.disconnect();
    node = null;
    if (ctx && ctx.state !== "closed") void ctx.close().catch(() => {});
    ctx = null;
    button.textContent = "Join";
    button.disabled = false;
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

/** Drop the session completely so Join starts a fresh one with the same phone id. */
function leave(why: string) {
  if (!joined) return;
  joined = false;
  clearTimeout(retryTimer);
  ws?.close();
  micStream?.getTracks().forEach((track) => track.stop());
  node?.disconnect();
  if (ctx && ctx.state !== "closed") void ctx.close().catch(() => {});
  ctx = null; node = null; micStream = null;
  const button = $<HTMLButtonElement>("join");
  button.textContent = "Join again";
  button.disabled = false;
  button.hidden = false;
  $("vuWrap").hidden = true;
  show({ tone: "bad", title: "Disconnected", detail: why });
}

/** Back on screen: restart audio and reconnect right away instead of waiting for the retry timer. */
async function comeBack() {
  const current = ctx;
  if (!joined || !current) return;
  pendingJump = true;
  void wake();
  // iOS can leave resume() pending forever, so don't wait on it for more than a moment.
  await Promise.race([current.resume().catch(() => {}), new Promise((r) => setTimeout(r, 1500))]);
  if (ctx !== current) return;
  if (current.state !== "running") { leave("The phone paused the microphone. Tap Join to connect again."); return; }
  if (!ws || ws.readyState >= WebSocket.CLOSING) { clearTimeout(retryTimer); retry = 0; connect(); }
  else show({ tone: "warn", title: "Needs re-sync", detail: "The screen was off, so timing was lost. Wait for the next chirp." });
}

document.addEventListener("visibilitychange", () => { if (document.hidden) pendingJump = true; else void comeBack(); });
$("join").addEventListener("click", () => void join());
