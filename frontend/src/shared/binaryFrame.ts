// 32-byte frame header, little-endian. Python: struct "<4sBBHIIQd" (shared/protocol.py).
export const FRAME_HEADER_SIZE = 32;
export const FLAG_CLOCK_JUMP = 0x01;
export const FLAG_CLIPPING = 0x02;

export interface FrameHeader {
  version?: number;
  flags: number;
  numSamples: number;
  seq: number;
  sampleRate: number;
  sampleIndex: number; // exact worklet sample count; safe below 2^53
  clientTimeMs: number;
}

export function encodeFrame(h: FrameHeader, pcm: Int16Array): ArrayBuffer {
  if (pcm.length !== h.numSamples) throw new Error("numSamples does not match pcm length");
  const buf = new ArrayBuffer(FRAME_HEADER_SIZE + 2 * pcm.length);
  const v = new DataView(buf);
  for (let i = 0; i < 4; i++) v.setUint8(i, "ASCS".charCodeAt(i));
  v.setUint8(4, h.version ?? 1);
  v.setUint8(5, h.flags & 0xff);
  v.setUint16(6, h.numSamples, true);
  v.setUint32(8, h.seq >>> 0, true);
  v.setUint32(12, h.sampleRate >>> 0, true);
  v.setBigUint64(16, BigInt(h.sampleIndex), true);
  v.setFloat64(24, h.clientTimeMs, true);
  new Int16Array(buf, FRAME_HEADER_SIZE, pcm.length).set(pcm);
  return buf;
}

export function decodeHeader(buf: ArrayBuffer): FrameHeader {
  const v = new DataView(buf);
  return {
    version: v.getUint8(4),
    flags: v.getUint8(5),
    numSamples: v.getUint16(6, true),
    seq: v.getUint32(8, true),
    sampleRate: v.getUint32(12, true),
    sampleIndex: Number(v.getBigUint64(16, true)),
    clientTimeMs: v.getFloat64(24, true),
  };
}
