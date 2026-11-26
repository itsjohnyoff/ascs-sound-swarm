# Analysis log

One line per real or simulated session: date, file, what was tested, the numbers.
Replay any line with `python -m analysis.spike data/recordings/<file>.npz`.

| date | file | what | result |
| --- | --- | --- | --- |
| 2026-11-24 | sim `sync_scenario(seed=7)` | 4 phones, 44.1/48 kHz, ±300 ms, ±50 ppm, echo +7 ms 0.65, 15 dB, 3 chirps 30 s apart | fit residual ≤ 0.0002 ms, alignment error 0.018 ms at +120 s |
| 2026-11-24 | spike-01 (real) | pending: H3 session, 2-4 phones, 3 chirps + 5 claps | run `python -m analysis.spike data/recordings/spike-01.npz` and paste here |
| 2026-11-25 | models/metrics.json | trained classifier (HGB, 12 features): 500/class synthetic + 6 DroneAudioSet drone-only recordings, split by recording | held-out macro F1 0.971 (DroneAudioSet 0.96, synthetic 0.98), sim judge 30/30, 14 ms/event, 477 kB |
| 2026-11-26 | sim `locate_scenario(seed=0..5)` | self-localization: 4 phones, laptop at room centre, palindrome chirp schedule, ±50 ppm drift, 44.1/48 kHz, echoes 0.3, 30 dB, 0-80 ms emission jitter | median position error 0.016 m (max 0.017 m), every phone "ok", ~150 ms solve |
| 2026-11-26 | `backend/dsp/sync.py` | real-room chirp fix (`min_ncc` 0.25->0.15, split-half NCC >= 0.5*min_ncc, vectorized `find_peaks` skipping claps, echo-safe <3-chirp GCC verify) | fixed 3/4 real phones failing calibration at 4 m (`ncc` 0.17-0.24) & clap-blocking on long recordings; `pytest tests/dsp` 53/53 pass, `sim.benchmark --seed 5 --occlude p2` unchanged at 0.095 m |
| 2026-11-26 | `20261-11-26-spike-01.npz` | Gate 0 (real): 4 phones (iOS 48k + 3 Android 48k/16k), 5x7 m room, 3 chirps + 5 claps | PASS: residual <= 0.5161 ms (`8c4n` 0.000, `tvb9` 0.422, `rjlg` 0.427, `3g8u` 0.516; Android inter-phone <= 0.01 ms), 0 gaps, 0.000% loss, 0.00% clipped, AGC/gate OK; 5/5 claps localized (res 0.20-0.52 ms, 11.4 ms/ev); real claps HGB 1/5 vs heuristic 5/5 |
