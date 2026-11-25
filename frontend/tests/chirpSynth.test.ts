import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { generateChirp, generateDualChirp } from "../src/shared/chirpSynth";

// Golden vectors generated from backend/dsp/chirp.py (same bytes as shared/chirp_*.f32 once Agent 1 commits them).
function golden(fs: number): Float32Array {
  const b = readFileSync(new URL(`./fixtures/chirp_${fs}.f32`, import.meta.url));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

describe("chirp synth", () => {
  for (const fs of [48000, 44100]) {
    it(`matches golden vector at ${fs} Hz within 1e-6`, () => {
      const g = golden(fs);
      const c = generateChirp(fs);
      expect(c.length).toBe(g.length);
      let max = 0;
      for (let i = 0; i < c.length; i++) max = Math.max(max, Math.abs(c[i]! - g[i]!));
      expect(max).toBeLessThan(1e-6);
    });
  }

  it("dual chirp places the second chirp at exactly 5.0 s", () => {
    const d = generateDualChirp(48000);
    const c = generateChirp(48000);
    expect(d.length).toBe(Math.round(48000 * 5.15));
    expect(d[240000 + 100]).toBe(c[100]);
    expect(d[100000]).toBe(0);
  });
});
