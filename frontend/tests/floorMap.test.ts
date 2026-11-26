import { describe, expect, it } from "vitest";
import { detectionNumber } from "../src/dashboard/floorMap";

describe("detectionNumber", () => {
  it("parses positive trailing event IDs", () => {
    expect(detectionNumber("evt-0001", 9)).toBe(1);
    expect(detectionNumber("evt-0012", 1)).toBe(12);
  });

  it("falls back when event ID lacks a positive number", () => {
    expect(detectionNumber("custom-id", 4)).toBe(4);
    expect(detectionNumber("evt-0000", 3)).toBe(3);
  });
});
