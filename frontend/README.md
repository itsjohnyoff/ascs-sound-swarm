# Frontend (Person 4)

Vanilla TS + Vite, three pages: `index.html` (dashboard + judge console + Gate 0 recorder), `phone.html`, `chirp.html`.

```bash
cd frontend && npm install && npm test && npm run build   # FastAPI serves dist/ itself
python scripts/start_https.py                              # HTTPS for phones (mic needs a secure context)
```

## Gate 0 run (real phones)
1. Phones on the table, tape-measure (x, y) of each phone mic and the laptop speaker.
2. Phones: open `https://<laptop>:8443/phone.html`, tap **Join**. Note the big phone id.
3. Laptop: open `/` and enter exact x/y per phone and the speaker in the side panels.
4. Click **Record** (name `spike-01`), open `/chirp.html` in a second tab.
5. Play a chirp at ~5 s, ~35 s, ~65 s; clap at a measured spot; click **Stop** (saves `data/recordings/spike-01.npz`).
6. `python -m backend.dsp.replay data/recordings/spike-01.npz` and read the held-out chirp #3 worst pairwise disagreement (PASS < 1.0 ms).

Chirp golden vectors live in `tests/fixtures/` (generated from `backend/dsp/chirp.py`); swap to `shared/chirp_*.f32` once Agent 1 commits them.
