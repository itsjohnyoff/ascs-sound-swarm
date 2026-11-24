# Analysis log

One line per real or simulated session: date, file, what was tested, the numbers.
Replay any line with `python -m analysis.spike data/recordings/<file>.npz`.

| date | file | what | result |
| --- | --- | --- | --- |
| 2026-11-24 | sim `sync_scenario(seed=7)` | 4 phones, 44.1/48 kHz, ±300 ms, ±50 ppm, echo +7 ms 0.65, 15 dB, 3 chirps 30 s apart | fit residual ≤ 0.0002 ms, alignment error 0.018 ms at +120 s |
| 2026-11-24 | spike-01 (real) | pending: H3 session, 2-4 phones, 3 chirps + 5 claps | run `python -m analysis.spike data/recordings/spike-01.npz` and paste here |
