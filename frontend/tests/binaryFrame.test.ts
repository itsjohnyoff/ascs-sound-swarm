import { describe, expect, it } from "vitest";
import { decodeHeader, encodeFrame } from "../src/shared/binaryFrame";

// Produced by Python: struct.pack("<4sBBHIIQd", b"ASCS", 1, 3, 2048, 7, 48000, 2**40 + 12345, 1732450201123.5)
const EXPECTED_HEX = "41534353010300080700000080bb000039300000000100000038e242e1357942";

const hex = (b: ArrayBuffer) => [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, "0")).join("");

describe("binary frame", () => {
  const header = {
    flags: 3, numSamples: 2048, seq: 7, sampleRate: 48000,
    sampleIndex: 2 ** 40 + 12345, clientTimeMs: 1732450201123.5,
  };
  const pcm = new Int16Array(2048).map((_, i) => i - 1024);

  it("header bytes match Python struct.pack", () => {
    const buf = encodeFrame(header, pcm);
    expect(buf.byteLength).toBe(32 + 4096);
    expect(hex(buf.slice(0, 32))).toBe(EXPECTED_HEX);
  });

  it("round-trips header and pcm", () => {
    const buf = encodeFrame(header, pcm);
    expect(decodeHeader(buf)).toMatchObject({ ...header, version: 1 });
    expect(new Int16Array(buf, 32, 2048)[5]).toBe(5 - 1024);
  });

  it("rejects length mismatch", () => {
    expect(() => encodeFrame({ ...header, numSamples: 10 }, pcm)).toThrow();
  });
});
