import type { Detection, Phone, Room } from "../shared/types";

export interface MapModel {
  room: Room; phones: Phone[]; detections: Detection[]; // newest last
  showHeat: boolean; showHyperbolas: boolean;
  groundTruth: { x: number; y: number } | null;
  linkedEstimate: Detection | null;
}
type DragTarget = { kind: "phone"; id: string } | { kind: "speaker" } | null;

const SYNC_COLOR: Record<string, string> = {
  synced: "#3fb950", drifting: "#d29922", unsynced: "#f85149", clock_jump: "#f85149",
};

export class FloorMap {
  private ctx: CanvasRenderingContext2D;
  private scale = 50; private ox = 0; private oy = 0;
  private drag: DragTarget = null;
  model: MapModel | null = null;
  judgeMode = false;
  onMovePhone: (id: string, x: number, y: number) => void = () => {};
  onMoveSpeaker: (x: number, y: number) => void = () => {};
  onPick: (x: number, y: number) => void = () => {};
  private dirty = true;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    new ResizeObserver(() => { this.dirty = true; }).observe(canvas);
    const loop = () => { if (this.dirty) { this.draw(); this.dirty = false; } requestAnimationFrame(loop); };
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
    if (Math.hypot(x - m.room.speaker_x_m, y - m.room.speaker_y_m) < tol) return { kind: "speaker" };
    for (const p of m.phones) if (Math.hypot(x - p.x_m, y - p.y_m) < tol) return { kind: "phone", id: p.phone_id };
    return null;
  }

  private down(e: PointerEvent) {
    const p = this.toRoom(e);
    if (this.judgeMode) { this.onPick(p.x, p.y); return; }
    this.drag = this.hit(p.x, p.y);
    if (this.drag) this.canvas.setPointerCapture(e.pointerId);
  }
  private move(e: PointerEvent) {
    if (!this.drag || !this.model) return;
    const p = this.toRoom(e);
    const x = Math.min(Math.max(p.x, 0), this.model.room.width_m);
    const y = Math.min(Math.max(p.y, 0), this.model.room.height_m);
    if (this.drag.kind === "speaker") { this.model.room.speaker_x_m = x; this.model.room.speaker_y_m = y; }
    else { const id = this.drag.id; const ph = this.model.phones.find((q) => q.phone_id === id); if (ph) { ph.x_m = x; ph.y_m = y; } }
    this.dirty = true;
  }
  private up(e: PointerEvent) {
    if (!this.drag || !this.model) return;
    const d = this.drag; this.drag = null;
    const p = this.toRoom(e);
    const x = Math.round(Math.min(Math.max(p.x, 0), this.model.room.width_m) * 100) / 100;
    const y = Math.round(Math.min(Math.max(p.y, 0), this.model.room.height_m) * 100) / 100;
    if (d.kind === "speaker") this.onMoveSpeaker(x, y); else this.onMovePhone(d.id, x, y);
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
        ctx.fillStyle = `rgba(255,140,0,${(v * v * 0.7).toFixed(3)})`;
        ctx.fillRect(this.ox + i * cw, this.oy + j * ch, cw + 0.5, ch + 0.5);
      }
    }

    ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1; ctx.fillStyle = "#6e7681"; ctx.font = "11px system-ui";
    for (let x = 0; x <= W + 1e-6; x++) { ctx.beginPath(); ctx.moveTo(this.px(x), this.py(0)); ctx.lineTo(this.px(x), this.py(H)); ctx.stroke(); ctx.fillText(`${x}`, this.px(x) - 3, this.py(0) - 6); }
    for (let y = 0; y <= H + 1e-6; y++) { ctx.beginPath(); ctx.moveTo(this.px(0), this.py(y)); ctx.lineTo(this.px(W), this.py(y)); ctx.stroke(); ctx.fillText(`${y}`, this.px(0) - 18, this.py(y) + 4); }
    ctx.strokeStyle = "#8b949e"; ctx.lineWidth = 2; ctx.strokeRect(this.px(0), this.py(0), W * this.scale, H * this.scale);

    if (m.showHyperbolas && last) this.drawHyperbolas(last);

    // detection trail (last 8) + latest pin with uncertainty
    const trail = m.detections.slice(-8);
    trail.forEach((d, i) => {
      const age = (i + 1) / trail.length;
      const col = d.sound_class === "drone" ? "137,87,229" : d.sound_class === "clap" ? "47,129,247" : "139,148,158";
      ctx.fillStyle = `rgba(${col},${(0.15 + 0.6 * age).toFixed(2)})`;
      ctx.beginPath(); ctx.arc(this.px(d.x_m), this.py(d.y_m), i === trail.length - 1 ? 9 : 5, 0, 7); ctx.fill();
      if (i === trail.length - 1) {
        ctx.strokeStyle = `rgba(${col},0.9)`; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.arc(this.px(d.x_m), this.py(d.y_m), Math.max(6, d.uncertainty_m * this.scale), 0, 7); ctx.stroke(); ctx.setLineDash([]);
      }
    });

    if (m.groundTruth) {
      const g = m.groundTruth;
      ctx.strokeStyle = "#3fb950"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(this.px(g.x) - 9, this.py(g.y)); ctx.lineTo(this.px(g.x) + 9, this.py(g.y)); ctx.moveTo(this.px(g.x), this.py(g.y) - 9); ctx.lineTo(this.px(g.x), this.py(g.y) + 9); ctx.stroke();
      const e = m.linkedEstimate;
      if (e) {
        ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(this.px(g.x), this.py(g.y)); ctx.lineTo(this.px(e.x_m), this.py(e.y_m)); ctx.stroke(); ctx.setLineDash([]);
        const err = Math.hypot(e.x_m - g.x, e.y_m - g.y);
        ctx.fillStyle = "#e8eef5"; ctx.font = "bold 14px system-ui";
        ctx.fillText(`${err.toFixed(2)} m`, (this.px(g.x) + this.px(e.x_m)) / 2 + 6, (this.py(g.y) + this.py(e.y_m)) / 2 - 6);
      }
    }

    // speaker
    ctx.fillStyle = "#e8eef5"; ctx.fillRect(this.px(m.room.speaker_x_m) - 7, this.py(m.room.speaker_y_m) - 7, 14, 14);
    ctx.font = "11px system-ui"; ctx.fillText("speaker", this.px(m.room.speaker_x_m) + 10, this.py(m.room.speaker_y_m) + 4);

    // phones: sync color dot + live RMS ring
    for (const p of m.phones) {
      const x = this.px(p.x_m), y = this.py(p.y_m);
      ctx.globalAlpha = p.connected ? 1 : 0.35;
      ctx.strokeStyle = "rgba(232,238,245,0.6)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 12 + Math.min(1, Math.sqrt(p.rms) * 3) * 16, 0, 7); ctx.stroke();
      ctx.fillStyle = SYNC_COLOR[p.sync_state] ?? "#f85149";
      ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fill();
      ctx.fillStyle = "#e8eef5"; ctx.font = "12px system-ui";
      ctx.fillText(p.label || p.phone_id, x + 14, y - 12);
      ctx.globalAlpha = 1;
    }
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
        ctx.strokeStyle = `rgba(210,153,34,${0.25 + 0.5 * t.weight})`;
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
