# Trained sound classifier

`drone_classifier.joblib` is loaded by `backend.classifier.model.SoundClassifier` when present
(`backend_name = "hgb"`). If it is missing or unreadable, the server uses the heuristic rules
(`backend_name = "heuristic"`). Delete the file to go back to heuristics.

| file | what |
| --- | --- |
| `drone_classifier.joblib` | `{"model": HistGradientBoostingClassifier, "features": [...], "classes": [...]}` |
| `metrics.json` | held-out scores (split by source recording), simulator judge check, training data counts |

## Retrain

```powershell
pip install -e ".[ml]"                                         # datasets + soundfile, once
python -m backend.classifier.train --synthetic 500 --droneaudioset 6   # what is shipped now (~2 min)
python -m backend.classifier.train                             # synthetic only, offline
```

Features: the 12 timbre features in `backend/classifier/features.py` (rms is excluded so level
never decides the class). Each phone is classified on its own and the 3 loudest are fused,
weighted by RMS.

Training data:
- synthetic: drones (single and multirotor), claps, and negatives (speech-like, knocks,
  coloured noise, whistles, melodies) from `sim/sources.py`, with random room impulse responses,
  phone-speaker playback (200 Hz to 6 kHz, soft clip), 5 to 8 kHz low-pass, and noise at 5 to 35 dB SNR (`sim/augment.py`).
- DroneAudioSet (`drone-only`, the first recording of 6 splits, first 60 s each, streamed, 16 kHz).
- venue recordings, if present: `data/train/<drone|clap|other|noise>/*.wav`. `clap` files can hold
  many claps (one window per onset). `noise` files are also mixed under the other classes as
  room background. **Record these at the venue** (20 s each: claps, room noise, drone audio from
  the demo speaker) and retrain. They matter more than the datasets.

> Gate 2 (real 10-position trial < 0.5 m) has not been recorded yet. The task says to train
> only if localization does not need help. The model is cheap to drop (delete the file).
