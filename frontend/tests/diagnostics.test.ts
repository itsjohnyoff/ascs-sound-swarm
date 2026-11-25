import { describe, expect, it } from "vitest";
import { Diagnostics } from "../src/phone/diagnostics";
import { FpsMeter } from "../src/dashboard/fps";
import { isDebug } from "../src/shared/debug";

describe("Diagnostics", () => {
  it("counts sampleIndex gaps only on discontinuities", () => {
    const d = new Diagnostics();
    d.onBatch(0, 2048); d.onBatch(2048, 2048); expect(d.gaps).toBe(0);
    d.onBatch(8192, 2048); // skipped 4096 samples
    expect(d.gaps).toBe(1); expect(d.gapSamples).toBe(4096);
    d.onBatch(10240, 2048); expect(d.gaps).toBe(1);
  });
  it("renders unknown track settings as ?", () => {
    expect(new Diagnostics().lines()[1]).toContain("AEC ?");
  });
});

describe("FpsMeter", () => {
  it("computes fps and cost over the window", () => {
    const m = new FpsMeter(1000);
    for (let i = 0; i <= 10; i++) m.record(i * 100, 2);
    expect(m.fps).toBeCloseTo(10); expect(m.avgCostMs).toBeCloseTo(2);
  });
  it("drops old samples", () => {
    const m = new FpsMeter(500);
    m.record(0, 1); m.record(100, 1); m.record(5000, 1); m.record(5100, 1);
    expect(m.fps).toBeCloseTo(10);
  });
});

describe("debug flag", () => {
  it("only on ?debug=1", () => {
    expect(isDebug("?debug=1")).toBe(true);
    expect(isDebug("")).toBe(false);
    expect(isDebug("?debug=0")).toBe(false);
  });
});
