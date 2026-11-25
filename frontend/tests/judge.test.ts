import { describe, expect, it } from "vitest";
import { insideHull, percentile, summarize } from "../src/dashboard/judge";
import { grade } from "../src/dashboard/calibration";

const corners = [{ x_m: 0.5, y_m: 0.5 }, { x_m: 7.5, y_m: 0.5 }, { x_m: 7.5, y_m: 5.5 }, { x_m: 0.5, y_m: 5.5 }];

describe("judge helpers", () => {
  it("hull inside/outside", () => {
    expect(insideHull(4, 3, corners)).toBe(true);
    expect(insideHull(0.1, 3, corners)).toBe(false);
    expect(insideHull(4, 3, corners.slice(0, 2))).toBe(false);
  });
  it("percentile matches numpy linear", () => {
    expect(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9)).toBeCloseTo(9.1);
    expect(percentile([], 0.5)).toBeNull();
  });
  it("summary", () => {
    const s = summarize([{ error_m: 0.2, missed: false }, { error_m: 0.4, missed: false }, { error_m: null, missed: true }] as never);
    expect(s.mean).toBeCloseTo(0.3); expect(s.missed).toBe(1);
  });
  it("sync grade thresholds", () => {
    expect(grade(0.5, true)).toBe("PASS"); expect(grade(2, true)).toBe("WARN");
    expect(grade(6, true)).toBe("FAIL"); expect(grade(0.1, false)).toBe("FAIL");
  });
});
