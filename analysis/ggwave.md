# ggwave vs LFM chirp as a timing anchor

Run: `python -m analysis.ggwave_eval --trials 10 --snr 20 10 0` (3 simulated phones, 48 kHz, 1.07 s ggwave "C01" vs our LFM chirp).

| SNR | ggwave decode | ggwave TDOA rms | LFM TDOA rms |
|---|---|---|---|
| 20 dB | 100% | 5.0 ms (173 cm) | 0.0001 ms |
| 10 dB | 100% | 11.3 ms (391 cm) | 0.0004 ms |
| 0 dB | 100% | 10.6 ms (364 cm) | 0.0013 ms |

- ggwave is good for carrying payload (IDs, room codes). It decodes reliably even at 0 dB.
- It is not a timing anchor: the decoder reports only when a frame finishes (1024 samples, 21 ms), after the end marker, so timing is frame-quantized and jittery. About 4 orders of magnitude worse than the matched-filter chirp.
- Gotcha: feeds smaller than 1024 samples are silently ignored, and decoder state leaks between streams, so use a fresh instance per stream.
- Decision: keep the LFM chirp for sync. Use ggwave only if we ever need to send data over sound.
