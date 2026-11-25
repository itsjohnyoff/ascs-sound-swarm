import { generateChirp, generateDualChirp, playBuffer } from "../shared/chirpSynth";

const out = document.getElementById("out")!;
let ctx: AudioContext | null = null;

async function play(dual: boolean) {
  ctx ??= new AudioContext();
  const fs = ctx.sampleRate;
  const t0 = new Date().toLocaleTimeString();
  out.textContent = `${t0}  playing ${dual ? "dual" : "single"} chirp at ${fs} Hz…`;
  await playBuffer(ctx, dual ? generateDualChirp(fs) : generateChirp(fs));
  out.textContent += "\ndone";
}

document.getElementById("one")!.addEventListener("click", () => void play(false));
document.getElementById("dual")!.addEventListener("click", () => void play(true));
