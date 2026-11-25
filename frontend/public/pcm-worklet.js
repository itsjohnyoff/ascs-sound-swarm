// AudioWorkletProcessor: counts every render quantum (even silent/missing input) so the
// sample timeline never skips, batches 2048 samples, posts Int16 PCM with transferred buffer.
const BATCH = 2048;

class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sampleIndex = 0; // monotonic count of samples seen so far
    this.fill = 0;
    this.startIndex = 0;
    this.buf = new Int16Array(BATCH);
    this.sumSq = 0;
    this.clipping = false;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    const n = ch ? ch.length : 128;
    for (let i = 0; i < n; i++) {
      const x = ch ? ch[i] : 0;
      if (this.fill === 0) this.startIndex = this.sampleIndex;
      const c = x > 1 ? 1 : x < -1 ? -1 : x;
      if (c >= 0.99 || c <= -0.99) this.clipping = true;
      this.sumSq += c * c;
      this.buf[this.fill++] = Math.round(c * 32767);
      this.sampleIndex++;
      if (this.fill === BATCH) this.flush();
    }
    return true;
  }

  flush() {
    const rms = Math.sqrt(this.sumSq / this.fill);
    this.port.postMessage(
      { pcm: this.buf, startSampleIndex: this.startIndex, rms, clipping: this.clipping },
      [this.buf.buffer],
    );
    this.buf = new Int16Array(BATCH);
    this.fill = 0;
    this.sumSq = 0;
    this.clipping = false;
  }
}

registerProcessor("pcm-worklet", PcmWorklet);
