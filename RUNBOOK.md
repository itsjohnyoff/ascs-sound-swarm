# Run book

## Setup (once)

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
cd frontend && npm ci && npm run build && cd ..
```

## Serve the demo (one origin, HTTPS)

```bash
python scripts/start_https.py --mode lan      # prints the join URL + QR; dashboard at /
python scripts/start_https.py --mode tunnel   # backup over cloudflared
```

Phones open `/phone.html`, tap **Join**, read their ID out loud. The dashboard is `/`, the laptop-speaker chirp page is `/chirp.html`.

## Without phones (one laptop)

```bash
uvicorn backend.server.app:app --port 8000                                 # serves frontend/dist
python -m sim.virtual_phones --server ws://127.0.0.1:8000 --phones 4 --scenario judge
python -m sim.benchmark --seed 0 [--occlude p2]                            # offline, writes analysis/benchmark-*.md
```

The virtual scenario plays chirps at t = 1 s and 11 s. Arm/solve around them (or keep auto-chirp on in the dashboard) or they show up as detections at the speaker.

Soak (Gate 3 prep, about 11 min): `python scripts/soak.py --minutes 10 --out soak.md`. It starts its own server on port 8765, streams 4 virtual phones with a chirp every 30 s (arm/solve like auto-chirp) and two claps/drones in between, then prints memory growth, dropped frames, per-chirp residual, detection rate, error and latency. Exit code 1 if a MUST fails. Add `--speed 4` for a quick check (memory and latency numbers are then not meaningful).

## Room setup

1. Phones near the corners, judge points inside their hull, never collinear (the GDOP banner warns).
2. Tape-measure each phone's bottom edge (mic) and the speaker to about 5 cm. Type them in the Phones and Room panels; the server keeps them, phones cannot overwrite them.
3. Speaker at a known spot with line of sight to every phone. Enter room temperature.
4. Network: phone hotspot or travel router. `--mode tunnel` only as a fallback.

## Sync spike (Gate 0, H3)

1. 2 to 4 phones (at least one iPhone, one Android) joined, positions entered.
2. `curl -X POST https://<host>:8443/api/record/start -H 'content-type: application/json' -d '{"name":"spike-01"}' -k`
3. Play 3 chirps 30 s apart from `/chirp.html`, then 5 claps at taped points.
4. `curl -X POST https://<host>:8443/api/record/stop -k` -> `data/recordings/spike-01.npz`.
5. Add the clap points to the file's meta if you want TDOA checks (`meta["clap_points"] = [{"x_m":..,"y_m":..}]`), then
   `python -m analysis.spike data/recordings/spike-01.npz` and paste the table into `analysis/LOG.md`.
   Gate 0 passes when every phone's residual is < 1 ms.

## Judged run

1. Calibrate from the dashboard; keep periodic chirps on (CONTRACTS 2).
2. Known-point check first: clap at a measured point, log it as the known point. It must be < 0.5 m.
3. For each of the 10 positions: set the true point, chirp before the trial, make the sound, pick the matching detection or mark it missed.
4. Export the CSV (`/api/eval/export.csv`).
