/** Rolling draw-rate + draw-time meter (?debug=1 only). */
export class FpsMeter {
  private stamps: number[] = [];
  private costs: number[] = [];
  constructor(private windowMs = 2000) {}

  record(now: number, costMs: number) {
    this.stamps.push(now); this.costs.push(costMs);
    while (this.stamps.length && now - this.stamps[0]! > this.windowMs) { this.stamps.shift(); this.costs.shift(); }
  }

  get fps(): number {
    if (this.stamps.length < 2) return 0;
    return ((this.stamps.length - 1) * 1000) / (this.stamps[this.stamps.length - 1]! - this.stamps[0]!);
  }
  get avgCostMs(): number { return this.costs.length ? this.costs.reduce((a, b) => a + b, 0) / this.costs.length : 0; }
  get maxCostMs(): number { return this.costs.length ? Math.max(...this.costs) : 0; }
}
