import type { Detection, Phone, Trial } from "../shared/types";

export interface HullPoint { x: number; y: number }

function cross(o: HullPoint, a: HullPoint, b: HullPoint) { return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x); }

export function convexHull(pts: HullPoint[]): HullPoint[] {
  const p = [...pts].sort((a, b) => a.x - b.x || a.y - b.y);
  if (p.length < 3) return p;
  const lo: HullPoint[] = [], up: HullPoint[] = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2]!, lo[lo.length - 1]!, q) <= 0) lo.pop(); lo.push(q); }
  for (const q of [...p].reverse()) { while (up.length >= 2 && cross(up[up.length - 2]!, up[up.length - 1]!, q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

export function insideHull(x: number, y: number, phones: Pick<Phone, "x_m" | "y_m">[]): boolean {
  const h = convexHull(phones.map((p) => ({ x: p.x_m, y: p.y_m })));
  if (h.length < 3) return false;
  return h.every((a, i) => cross(a, h[(i + 1) % h.length]!, { x: x, y: y }) >= -1e-9);
}

export function errorM(d: Detection, gx: number, gy: number) { return Math.hypot(d.x_m - gx, d.y_m - gy); }

export function percentile(xs: number[], q: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b); const pos = (s.length - 1) * q; const lo = Math.floor(pos);
  return s[lo]! + (s[Math.min(lo + 1, s.length - 1)]! - s[lo]!) * (pos - lo);
}

export function summarize(trials: Trial[]) {
  const errs = trials.filter((t) => t.error_m != null).map((t) => t.error_m!);
  return {
    total: trials.length, missed: trials.filter((t) => t.missed).length,
    mean: errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : null,
    median: percentile(errs, 0.5), p90: percentile(errs, 0.9),
  };
}

export function trialsCsvHint() { return "trial_idx,label,expected_class,gt_x_m,gt_y_m,est_x_m,est_y_m,error_m,missed,pred_class,class_confidence,residual_ms,timestamp_s"; }
