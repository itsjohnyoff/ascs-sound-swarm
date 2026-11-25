export interface TrackInfo { aec?: boolean; ns?: boolean; agc?: boolean; channels?: number }

/** Local counters read off the screen during real-device tests (?debug=1 only). */
export class Diagnostics {
  sampleRate = 0;
  track: TrackInfo = {};
  framesSent = 0;
  framesDropped = 0; // worklet produced a frame while the socket was closed
  gaps = 0; // sampleIndex discontinuities seen locally
  gapSamples = 0;
  reconnects = 0;
  clockJumps = 0;
  private expected: number | null = null;

  /** Call for every worklet batch, whether or not it was sent. */
  onBatch(startSampleIndex: number, numSamples: number) {
    if (this.expected !== null && startSampleIndex !== this.expected) {
      this.gaps++;
      this.gapSamples += Math.abs(startSampleIndex - this.expected);
    }
    this.expected = startSampleIndex + numSamples;
  }

  lines(): string[] {
    const t = this.track;
    const on = (v?: boolean) => (v === undefined ? "?" : v ? "ON" : "off");
    return [
      `sampleRate ${this.sampleRate} Hz`,
      `AEC ${on(t.aec)}  NS ${on(t.ns)}  AGC ${on(t.agc)}  ch ${t.channels ?? "?"}`,
      `frames sent ${this.framesSent}  not sent ${this.framesDropped}`,
      `index gaps ${this.gaps} (${this.gapSamples} samples)`,
      `reconnects ${this.reconnects}  clock jumps ${this.clockJumps}`,
    ];
  }
}
