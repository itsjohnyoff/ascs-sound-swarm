# Venue Audio Capture Protocol

Operator runbook for the real-phone Gate 0 sync spike and venue classifier evaluation. Use `YYYY-MM-DD` for `<date>` (e.g. `2026-11-26`).

## 1. Pre-flight & Gate 0 Spike (`data/recordings/spike-01.npz`)

- **Phones:** 2–4 real phones placed around the demo perimeter at measured `(x_m, y_m)` coordinates.
- **Browser audio constraints:** verify every phone captures with `echoCancellation: false`, `noiseSuppression: false`, `autoGainControl: false`.
- **Capture:** start recording to `data/recordings/spike-01.npz`, play **3 chirps** (spaced ~10–30 s apart), then make **5 claps** at taped `(x_m, y_m)` ground-truth points inside the phone polygon.
- **Analyze:**

```bash
python -m analysis.spike data/recordings/spike-01.npz
```

- **Gate 0 pass thresholds (every phone):**
  - Sync fit residual `< 1 ms`
  - `0` dropped-frame gaps
  - Clipped samples `< 1%`
  - `AGC/gate` reports `OK` (no noise gating or post-clap AGC ducking)

If any phone fails Gate 0, fix the mic/browser constraint or replace that phone before recording classifier clips.

## 2. Venue Classifier Clips

Record three separate `.npz` files with the same phone layout. Keep calibration chirps in the separate Gate 0 recording and turn off **auto-chirp 30 s** in the dashboard for these clips. The evaluator asks for a fresh recording if it detects a calibration chirp on the selected phone.

| File | Duration | Protocol |
| --- | --- | --- |
| `data/recordings/<date>-venue-claps.npz` | 20 s min (25 s recommended, 2 s buffer) | Clap ~1 time every 1.5–2 s for 10+ claps across taped points inside the phone polygon. |
| `data/recordings/<date>-venue-noise.npz` | 20 s min (25 s recommended, 2 s buffer) | Normal venue ambient noise (chatter, HVAC, footsteps); no claps and no drone playback. |
| `data/recordings/<date>-venue-drone.npz` | 20 s min (25 s recommended, 2 s buffer) | Play drone audio through the **actual demo speaker** at >= 2 positions inside the phone polygon, 1.5–4 m from phones, at realistic demo volume without clipping. |

## 3. Evaluation & Live-Model Decision

Run the offline heuristic-vs-trained comparison on the three venue recordings:

```bash
python -m analysis.classifier_eval \
  data/recordings/<date>-venue-claps.npz \
  data/recordings/<date>-venue-noise.npz \
  data/recordings/<date>-venue-drone.npz
```

- **Target:** macro F1 `> 0.9` on venue clips.
- **Decision rule:** run the comparison on venue clips alone. The evaluator uses the loudest phone in each clip because the tracks are uncalibrated. If any class has no usable examples, recapture it before choosing a backend.
  - If the trained model wins, keep `models/drone_classifier.joblib`.
  - If the heuristic wins, move `models/drone_classifier.joblib` out of `models/` and restart the server to load the heuristic backend.

## 4. Exact `analysis/LOG.md` Lines

Append both rows to `analysis/LOG.md` (columns: `| date | file | what | result |`):

```markdown
| <date> | spike-01.npz | Gate 0: <N> phones, 3 chirps + 5 claps at taped points | residual <= <X.XX> ms, <0> gaps, clipped < <X.X>%, AGC/gate <OK/FLAG> |
| <date> | <date>-venue-{claps,noise,drone}.npz | venue classifier eval (<N> events, demo speaker >=2 pos 1.5-4 m) | heuristic macro F1 <0.XX> (<X> ms/ev) vs trained macro F1 <0.XX> (<X> ms/ev); live=<heuristic/trained> |
```

## 5. Repo Hygiene

- Never commit `.npz` recordings `> 50 MB`.
- If venue macro F1 is below target and retraining is needed, export labeled clips as WAVs to `data/train/<drone|clap|other|noise>/*.wav`, retrain, and keep `models/drone_classifier.joblib` `< 10 MB`.
