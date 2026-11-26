import type { Detection, Phone, Room } from "../shared/types";
import { gradeResidual } from "./readiness";
import { isRoomPoint } from "./trialFlow";

export interface MapModel {
  room: Room; phones: Phone[]; detections: Detection[]; // newest last
  showHeat: boolean; showHyperbolas: boolean; showDistances: boolean;
  groundTruth: { x: number; y: number } | null;
  linkedEstimate: Detection | null;
}
type DragTarget = { id: string } | null; // only phones move: the speaker is fixed at the centre

const INK = "#dde4ea", MUTE = "#8996a1", GRID = "#1b222a", EDGE = "#5d6b79";
const GRADE_COLOR = { pass: "#66b07a", warn: "#cba653", fail: "#d4706a" } as const;
const CLASS_RGB: Record<string, string> = { drone: "165,139,214", clap: "90,143,208", other: "137,150,161" };

export class FloorMap {
  private ctx: CanvasRenderingContext2D;
  private scale = 50; private ox = 0; private oy = 0;
  private drag: DragTarget = null;
  model: MapModel | null = null;
  judgeMode = false;
  onMovePhone: (id: string, x: number, y: number) => void = () => {};
  onPick: (x: number, y: number) => void = () => {};
  private dirty = true;
  /** Called after each draw with wall time and cost in ms (used by the ?debug=1 fps meter). */
  onDraw: (now: number, costMs: number) => void = () => {};

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    canvas.addEventListener("pointercancel", () => { this.drag = null; });
    new ResizeObserver(() => { this.dirty = true; }).observe(canvas);
    const loop = () => {
      if (this.dirty) { const t0 = performance.now(); this.draw(); this.dirty = false; this.onDraw(t0, performance.now() - t0); }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  invalidate() { this.dirty = true; }

  toRoom(e: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left - this.ox) / this.scale, y: (e.clientY - r.top - this.oy) / this.scale };
  }

  private hit(x: number, y: number): DragTarget {
    const m = this.model; if (!m) return null;
    const tol = 14 / this.scale;
    for (const p of m.phones) if (Math.hypot(x - p.x_m, y - p.y_m) < tol) return { id: p.phone_id };
    return null;
  }

  private down(e: PointerEvent) {
    const p = this.toRoom(e);
    if (this.judgeMode) return;
    this.drag = this.hit(p.x, p.y);
    if (this.drag) this.canvas.setPointerCapture(e.pointerId);
  }
  private move(e: PointerEvent) {
    if (!this.drag || !this.model) return;
    const p = this.toRoom(e);
    const x = Math.min(Math.max(p.x, 0), this.model.room.width_m);
    const y = Math.min(Math.max(p.y, 0), this.model.room.height_m);
    const id = this.drag.id; const ph = this.model.phones.find((q) => q.phone_id === id); if (ph) { ph.x_m = x; ph.y_m = y; }
    this.dirty = true;
  }
  private up(e: PointerEvent) {
    if (!this.model) return;
    if (this.judgeMode) {
      const p = this.toRoom(e);
      if (isRoomPoint(this.model.room, p.x, p.y)) this.onPick(p.x, p.y);
      return;
    }
    if (!this.drag) return;
    const d = this.drag; this.drag = null;
    const p = this.toRoom(e);
    const x = Math.round(Math.min(Math.max(p.x, 0), this.model.room.width_m) * 100) / 100;
    const y = Math.round(Math.min(Math.max(p.y, 0), this.model.room.height_m) * 100) / 100;
    this.onMovePhone(d.id, x, y);
  }

  private px(x: number) { return this.ox + x * this.scale; }
  private py(y: number) { return this.oy + y * this.scale; }

  private draw() {
    const c = this.canvas, ctx = this.ctx, m = this.model;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!m) return;
    const pad = 30;
    this.scale = Math.min((w - 2 * pad) / m.room.width_m, (h - 2 * pad) / m.room.height_m);
    this.ox = (w - m.room.width_m * this.scale) / 2; this.oy = (h - m.room.height_m * this.scale) / 2;
    const W = m.room.width_m, H = m.room.height_m;

    const last = m.detections[m.detections.length - 1];
    if (m.showHeat && last?.heatmap) {
      const { nx, ny, values } = last.heatmap;
      const cw = (W * this.scale) / nx, ch = (H * this.scale) / ny;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const v = values[j * nx + i] ?? 0;
        ctx.fillStyle = `rgba(203,166,83,${(v * v * 0.6).toFixed(3)})`;
        ctx.fillRect(this.ox + i * cw, this.oy + j * ch, cw + 0.5, ch + 0.5);
      }
    }

    ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.fillStyle = MUTE; ctx.font = "12px system-ui";
    for (let x = 0; x <= W + 1e-6; x++) { ctx.beginPath(); ctx.moveTo(this.px(x), this.py(0)); ctx.lineTo(this.px(x), this.py(H)); ctx.stroke(); ctx.fillText(`${x}`, this.px(x) - 3, this.py(0) - 6); }
    for (let y = 0; y <= H + 1e-6; y++) { ctx.beginPath(); ctx.moveTo(this.px(0), this.py(y)); ctx.lineTo(this.px(W), this.py(y)); ctx.stroke(); ctx.fillText(`${y}`, this.px(0) - 18, this.py(y) + 4); }
    ctx.strokeStyle = EDGE; ctx.lineWidth = 1.5; ctx.strokeRect(this.px(0), this.py(0), W * this.scale, H * this.scale);

    if (m.showHyperbolas && last) this.drawHyperbolas(last);

    // detection trail (last 8) + latest pin with uncertainty
    const trail = m.detections.slice(-8);
    trail.forEach((d, i) => {
      const age = (i + 1) / trail.length;
      const col = CLASS_RGB[d.sound_class] ?? CLASS_RGB.other!;
      ctx.fillStyle = `rgba(${col},${(0.15 + 0.6 * age).toFixed(2)})`;
      ctx.beginPath(); ctx.arc(this.px(d.x_m), this.py(d.y_m), i === trail.length - 1 ? 9 : 5, 0, 7); ctx.fill();
      if (i === trail.length - 1) {
        ctx.strokeStyle = `rgba(${col},0.9)`; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.arc(this.px(d.x_m), this.py(d.y_m), Math.max(6, d.uncertainty_m * this.scale), 0, 7); ctx.stroke(); ctx.setLineDash([]);
      }
    });

    if (m.groundTruth) {
      const g = m.groundTruth;
      ctx.strokeStyle = GRADE_COLOR.pass; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(this.px(g.x) - 9, this.py(g.y)); ctx.lineTo(this.px(g.x) + 9, this.py(g.y)); ctx.moveTo(this.px(g.x), this.py(g.y) - 9); ctx.lineTo(this.px(g.x), this.py(g.y) + 9); ctx.stroke();
      const e = m.linkedEstimate;
      if (e) {
        ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(this.px(g.x), this.py(g.y)); ctx.lineTo(this.px(e.x_m), this.py(e.y_m)); ctx.stroke(); ctx.setLineDash([]);
        const err = Math.hypot(e.x_m - g.x, e.y_m - g.y);
        ctx.fillStyle = INK; ctx.font = "bold 16px system-ui";
        ctx.fillText(`${err.toFixed(2)} m`, (this.px(g.x) + this.px(e.x_m)) / 2 + 6, (this.py(g.y) + this.py(e.y_m)) / 2 - 6);
      }
    }

    if (m.showDistances && !this.judgeMode) this.drawDistances(m);

    // speaker
    ctx.fillStyle = INK; ctx.fillRect(this.px(m.room.speaker_x_m) - 7, this.py(m.room.speaker_y_m) - 7, 14, 14);
    ctx.font = "13px system-ui"; ctx.fillText("Speaker", this.px(m.room.speaker_x_m) + 10, this.py(m.room.speaker_y_m) + 4);

    // phones: grade-colored dot, live level ring, name and timing error beside it
    for (const p of m.phones) {
      const x = this.px(p.x_m), y = this.py(p.y_m);
      const synced = p.sync_state === "synced" || p.sync_state === "drifting";
      const grade = gradeResidual(p.residual_ms, synced);
      ctx.globalAlpha = p.connected ? 1 : 0.35;
      ctx.strokeStyle = "rgba(232,238,245,0.55)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 13 + Math.min(1, Math.sqrt(p.rms) * 3) * 16, 0, 7); ctx.stroke();
      ctx.fillStyle = GRADE_COLOR[grade];
      ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fill();
      ctx.fillStyle = INK; ctx.font = "600 13px system-ui";
      ctx.fillText(p.label || p.phone_id, x + 16, y - 4);
      ctx.fillStyle = MUTE; ctx.font = "12px system-ui";
      ctx.fillText(!p.connected ? "offline" : synced && p.residual_ms != null ? `${p.residual_ms.toFixed(2)} ms` : "not synced", x + 16, y + 11);
      ctx.globalAlpha = 1;
    }
  }

  // Dashed lines with lengths: speaker to each phone, and around the outline of the phones.
  private drawDistances(m: MapModel) {
    const ctx = this.ctx;
    const speaker = { x_m: m.room.speaker_x_m, y_m: m.room.speaker_y_m };
    const angle = (p: Phone) => Math.atan2(p.y_m - speaker.y_m, p.x_m - speaker.x_m);
    const ring = [...m.phones].sort((a, b) => angle(a) - angle(b));
    const segments: [{ x_m: number; y_m: number }, { x_m: number; y_m: number }][] = m.phones.map((p) => [speaker, p]);
    if (ring.length === 2) segments.push([ring[0]!, ring[1]!]);
    if (ring.length > 2) ring.forEach((p, i) => segments.push([p, ring[(i + 1) % ring.length]!]));
    ctx.font = "12px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const [a, b] of segments) {
      ctx.strokeStyle = "rgba(137,150,161,0.45)"; ctx.lineWidth = 1; ctx.setLineDash([4, 5]);
      ctx.beginPath(); ctx.moveTo(this.px(a.x_m), this.py(a.y_m)); ctx.lineTo(this.px(b.x_m), this.py(b.y_m)); ctx.stroke(); ctx.setLineDash([]);
      const text = `${Math.hypot(a.x_m - b.x_m, a.y_m - b.y_m).toFixed(2)} m`;
      const at = (t: number) => [this.px(a.x_m + (b.x_m - a.x_m) * t), this.py(a.y_m + (b.y_m - a.y_m) * t)] as const;
      let [tx, ty] = at(0.5);
      // A line between two phones can cross the middle: keep its label off the speaker.
      if (a !== speaker && Math.hypot(tx - this.px(speaker.x_m), ty - this.py(speaker.y_m)) < 70) {
        const dx = b.x_m - a.x_m, dy = b.y_m - a.y_m, len = Math.hypot(dx, dy) || 1;
        const side = dy >= 0 ? 1 : -1; // push down and to the left, away from the "Speaker" text
        tx += (-dy / len) * 24 * side; ty += (dx / len) * 24 * side;
      }
      const tw = ctx.measureText(text).width + 8;
      ctx.fillStyle = "#12171d"; ctx.fillRect(tx - tw / 2, ty - 8, tw, 16);
      ctx.fillStyle = MUTE; ctx.fillText(text, tx, ty);
    }
    ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
  }

  // Each inlier TDOA gives a hyperbola |d_i - d_j| = c * tdoa. Sampled in polar-ish form via a coarse grid march.
  private drawHyperbolas(d: Detection) {
    const m = this.model!; const C = 331.3 + 0.606 * m.room.temperature_c;
    const byId = new Map(m.phones.map((p) => [p.phone_id, p]));
    {
      const ctx = this.ctx; ctx.lineWidth = 1.5;
      for (const t of d.pair_tdoas_ms) {
        if (!t.inlier) continue;
        const a = byId.get(t.i), b = byId.get(t.j); if (!a || !b) continue;
        const target = (t.tdoa_ms / 1000) * C; // d_i - d_j (sign per contract: tdoa = t_i - t_j)
        ctx.strokeStyle = `rgba(210,153,34,${0.3 + 0.5 * t.weight})`;
        const step = 0.05;
        ctx.beginPath();
        for (let x = 0; x <= m.room.width_m; x += step) for (let y = 0; y <= m.room.height_m; y += step) {
          const f = Math.hypot(x - a.x_m, y - a.y_m) - Math.hypot(x - b.x_m, y - b.y_m) - target;
          if (Math.abs(f) < step * 0.5) { ctx.moveTo(this.px(x), this.py(y)); ctx.lineTo(this.px(x) + 1, this.py(y) + 1); }
        }
        ctx.stroke();
      }
    }
  }
}
