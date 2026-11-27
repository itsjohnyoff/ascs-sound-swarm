# Run book

## Quick start

1. Start the server: `python scripts/start_https.py --mode lan`. If it warns that this Wi-Fi may block phones, use a phone hotspot.
2. Put the laptop in the middle of the room. Open the dashboard at `https://localhost:8443/`, enter the room size and press **Apply**.
3. On each phone, scan **Phone QR**, tap **Join** and turn the volume up. Put the phones near the corners.
4. Press **Get ready**. It checks the phones joined, how loud the laptop is, finds where the phones are by sound, syncs their clocks, and asks you to clap next to one phone. If a step fails it says what to fix; press **Continue**.
5. Open **Judge mode** (or open `/?judge=1` on the projector laptop).

Everything else (manual positions, map layers, calibration table, recording) is under **Details**.

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

Phones open `/phone.html`, tap **Join**, read their ID out loud. The dashboard is `/`, the laptop-speaker chirp page is `/chirp.html`, and `/api/health` reports connected/synced phones and active classifier backend.

- **Gate preflight:** `python -m scripts.phone_gate preflight https://localhost:8443 --insecure` prints the join URL, count, actual sample rates and sync state for each connected phone. Add `--require-sync` after calibration, or `--min-phones 2` for Gate 0.
- **Real-phone sessions:** follow the [gate plan](docs/superpowers/plans/2026-11-25-a1-real-phone-gates.md) for capture names, evidence, mock runs and the release freeze.
- **LAN TLS trust:** without `mkcert`, phones see a self-signed cert warning once per IP — on iOS Safari tap *Show Details → visit this website*; on Android Chrome tap *Advanced → Proceed*.
- **Network fallback:** prefer a phone hotspot or travel router so venue Wi-Fi AP isolation cannot block phones from reaching the laptop. Use `--mode tunnel` (`cloudflared`) only if LAN is blocked.

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
2. Phones find their own positions by sound (**Get ready**, or **Details → Locate phones again**). If the map comes out turned or mirrored compared with the room, use **Turn** or **Mirror**. A phone that couldn't be placed can be dragged or typed in under **Details**; the server keeps positions, phones cannot overwrite them.
3. The speaker (laptop) is always the room centre, with line of sight to every phone. Enter room temperature.
4. Before the judged run, **Check phones haven't moved** (judge panel, under Checks and export) flags any phone moved more than 10 cm.
5. Network: phone hotspot or travel router. `--mode tunnel` only as a fallback.

## Sync spike (Gate 0, H3)

1. 2 to 4 phones (at least one iPhone, one Android) joined, positions entered.
2. `curl -X POST https://<host>:8443/api/record/start -H 'content-type: application/json' -d '{"name":"spike-01"}' -k`
3. Play 3 chirps 30 s apart from `/chirp.html`, then 5 claps at taped points.
4. `curl -X POST https://<host>:8443/api/record/stop -k` (or `?name=spike-01`) -> `data/recordings/spike-01.npz`.
5. Save taped clap points in order as a JSON array of `{"x_m": ..., "y_m": ...}` objects. Run
   `python -m analysis.spike data/recordings/spike-01.npz --clap-points data/recordings/spike-01-clap-points.json` and log the results in `analysis/LOG.md`.
   Gate 0 passes when every phone's residual is < 1 ms, 0 gaps, clipped < 1%, and `AGC/gate` is `OK`. See `analysis/venue_capture.md` for the full Gate 0 + venue classifier capture protocol.

## Judged run (Gate 2 & demo)

1. Pre-flight: verify `frontend/dist` is built (`npm run build`), all 4+ phones are green (`synced`), no GDOP warning, and room temperature is set.
2. Start a recording (`POST /api/record/start` with `{"name": "<YYYY-MM-DD>-gate2"}`) so the session can be replayed and tuned afterwards.
3. Calibrate from the dashboard; keep periodic auto-chirp (30 s) on (CONTRACTS 2).
4. Known-point check first: clap at a measured point, log it as the known point (`POST /api/eval/known-point`). It must be < 0.5 m before judges begin.
5. For each of the 10 positions: set the true point, chirp before the trial, make the sound (clap or drone), pick the matching detection from the last 5 detections or mark it missed.
6. Export the CSV (`/api/eval/export.csv`), stop the recording (`POST /api/record/stop`), and log the mean/median/p90 error and recording path in `analysis/LOG.md`.

## Drift soak (Gate 3)

- **Virtual:** `python scripts/soak.py --minutes 10 --out analysis/soak-10m.md` (MUST pass: no crash, no phone disconnects, memory growth <= 50 MB, every chirp after the first calibrates, detection rate >= 90%).
- **Real phones:** keep 4 phones connected for 10 minutes with auto-chirp on, record to `data/recordings/<YYYY-MM-DD>-gate3-soak.npz`, check dropped frames (< 1% target) and residuals (< 1 ms) without manual re-sync, and log in `analysis/LOG.md`.

For each live capture, run `python -m scripts.recording_summary data/recordings/<name>.npz` to see loss and continuity per phone. Add `--append-log --date YYYY-MM-DD --what "Gate 3" --result "measured results"` to append a session row. Run `python -m scripts.phone_gate timing data/recordings/<name>.npz` for per-solve calibration and transient broadcast timing. Use a video with the clap and dashboard in the same recording to measure the full clap-to-map interval.

## Troubleshooting

- **Phone disconnects or clock-jumps:** the server automatically re-references if the reference phone drops (when >= 2 synced phones remain) and marks the affected phone `clock_jump` / `unsynced`. Let the next auto-chirp (or click Calibrate) bring it back.
- **GDOP warning:** spread phones around the room corners so they are not collinear and keep judge points inside their convex hull.
- **No detections:** confirm >= 3 phones are `synced` on `/api/health`, check RMS bars on the dashboard, and verify no phone is stuck clipping.

### Tune a recorded run

Add a `t_s` column to the judge export and mark each sound's onset in seconds from the first sample of the first phone track by sorted phone ID. Use the same track for every trial, including trials marked missed, and keep that phone recording and synchronized throughout the run. The exported `timestamp_s` records when a result was published or marked missed.

To read onset times in an audio editor, export that phone's waveform with Python.

```python
from backend.dsp.replay import load_recording_npz
from scipy.io.wavfile import write

recording = load_recording_npz("data/recordings/<name>.npz")
phone = recording.slices[min(recording.slices)]
write("judge-reference.wav", phone.sample_rate, phone.samples_f32)
```

Run `python -m backend.dsp.tune data/recordings/<name>.npz --truth judge-eval.csv`.
For an untimed list of positions, a sweep row can only score errors when the detection and trial counts agree. Rows with unequal counts report every trial as unmatched. Use timed annotations if a recording may contain extra detections or missed trials.
