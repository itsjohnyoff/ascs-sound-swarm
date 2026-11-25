// Must match shared/protocol.py + backend/dsp/chirp.py (golden vectors in tests/fixtures).
export const CHIRP_F0_HZ = 1500;
export const CHIRP_F1_HZ = 8000;
export const CHIRP_DURATION_S = 0.15;
export const CHIRP_TUKEY_ALPHA = 0.1;
export const CHIRP_AMPLITUDE = 0.85;
export const DUAL_CHIRP_GAP_S = 5.0;

export function generateChirp(fs: number): Float32Array {
  const n = Math.round(fs * CHIRP_DURATION_S);
  const out = new Float32Array(n);
  const rate = (CHIRP_F1_HZ - CHIRP_F0_HZ) / CHIRP_DURATION_S;
  const a = CHIRP_TUKEY_ALPHA;
  for (let k = 0; k < n; k++) {
    const t = k / fs;
    const u = k / (n - 1);
    let w = 1;
    if (u < a / 2) w = 0.5 * (1 - Math.cos((2 * Math.PI * u) / a));
    else if (u > 1 - a / 2) w = 0.5 * (1 - Math.cos((2 * Math.PI * (1 - u)) / a));
    const phase = 2 * Math.PI * (CHIRP_F0_HZ * t + 0.5 * rate * t * t);
    out[k] = CHIRP_AMPLITUDE * w * Math.sin(phase);
  }
  return out;
}

export function generateDualChirp(fs: number): Float32Array {
  const single = generateChirp(fs);
  const gap = Math.round(fs * DUAL_CHIRP_GAP_S);
  const out = new Float32Array(gap + single.length);
  out.set(single, 0);
  out.set(single, gap);
  return out;
}

/** Play a buffer through the default output; resolves when playback ends. */
export async function playBuffer(ctx: AudioContext, samples: Float32Array): Promise<void> {
  if (ctx.state !== "running") await ctx.resume();
  const buffer = ctx.createBuffer(1, samples.length, ctx.sampleRate);
  buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  await new Promise<void>((resolve) => {
    src.onended = () => resolve();
    src.start();
    setTimeout(resolve, (samples.length / ctx.sampleRate) * 1000 + 1000); // never hang if onended is missed
  });
}
