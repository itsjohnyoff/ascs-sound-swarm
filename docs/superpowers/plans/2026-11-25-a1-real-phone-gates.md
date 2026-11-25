# Real-phone gate plan

Run in order with a laptop, speaker, taped positions, at least one iPhone and one Android phone, and four phones for Gates 1–3. Use the venue network for the primary measurements. Thresholds below come from the Agent 1 brief, [run book](../../../RUNBOOK.md), and [venue capture protocol](../../../analysis/venue_capture.md).

## Start the server and check the room

From the repository root, prepare the laptop and leave the HTTPS launcher running.

```bash
source .venv/bin/activate
npm ci --prefix frontend
npm run build --prefix frontend
python scripts/start_https.py --mode lan
```

Use the printed join URL on the phones. Open the dashboard on the server laptop. Set the room dimensions, temperature, speaker location and each phone mic position from the tape measurements. Complete the device checks in [DEVICE_TEST.md](../../../frontend/DEVICE_TEST.md) on iOS Safari and Android Chrome. Use `?debug=1` on the phone pages to record the actual rate, applied microphone settings and clock jumps.

In a second shell at the repository root:

```bash
source .venv/bin/activate
set -o pipefail
BASE=https://localhost:8443
DAY=$(date +%F)
OUT=data/recordings
mkdir -p "$OUT"
demo() { curl --insecure --fail-with-body --silent --show-error "$@"; }
preflight() { python -m scripts.phone_gate preflight "$BASE" --insecure "$@"; }
capture_start() {
  demo -X POST -H 'Content-Type: application/json' --data "{\"name\":\"$1\"}" "$BASE/api/record/start"
}
capture_stop() { demo -X POST "$BASE/api/record/stop"; }
```

Use a fresh recording name if you repeat a capture (for example, add `-retry1`). Keep the original evidence. If you change to a trusted tunnel origin, change `BASE` and remove the insecure options.

## Gate 0 · Sync spike

```bash
NAME="$DAY-spike-01"
preflight --min-phones 2
capture_start "$NAME"
```

Use two to four phones and turn off dashboard auto-chirp. From `/chirp.html`, play three single chirps 30 seconds apart. After the third chirp, make five claps at the taped positions, with a few seconds between claps. Write the five points in clap order to `data/recordings/<date>-spike-01-clap-points.json` as a JSON array of objects with `x_m` and `y_m` in meters.

```bash
capture_stop
python -m analysis.spike "$OUT/$NAME.npz" --clap-points "$OUT/$NAME-clap-points.json" \
  | tee "$OUT/$NAME-spike.txt"
python -m scripts.recording_summary "$OUT/$NAME.npz"
```

**Artifacts:** `<date>-spike-01.npz`, `<date>-spike-01-clap-points.json`, and `<date>-spike-01-spike.txt` under `data/recordings/`. Append one row to `analysis/LOG.md` with the pass/fail decision, per-phone sample rate, fit residual, gap count, clipping and AGC/gate result.

**Pass:** Fit all three chirps on two to four phones. Every phone has residual below 1 ms, zero dropped frames or sample-index gaps, continuous capture, less than 1% clipped samples and an OK AGC/gate check. **Fail:** Any phone misses a threshold or leaves the evidence incomplete. Stop after Gate 0 and review the individual residuals together. If any residual is 1 ms or higher or there are sample-index gaps, stop gate work and debug capture before attempting Gate 1.

## Gate 1 · Four phones and one pin

```bash
NAME="$DAY-gate1"
preflight
capture_start "$NAME"
```

Calibrate from the dashboard, wait for the solve to finish, and run the sync check below. Wait at least one second before clapping at a taped point inside the phone layout. Observe the pin. Film the clap and mapped pin with one camera for acoustic onset to painted pin timing. Measure elapsed time on the video using its timestamps or frame rate.

```bash
preflight --require-sync
# Clap, verify the mapped pin, then save the evidence.
demo "$BASE/api/detections/recent" > "$OUT/$NAME-detections.json"
capture_stop
python -m scripts.phone_gate timing "$OUT/$NAME.npz" | tee "$OUT/$NAME-timing.txt"
python -m scripts.recording_summary "$OUT/$NAME.npz"
```

**Artifacts:** `<date>-gate1.npz`, `<date>-gate1-detections.json`, `<date>-gate1-timing.txt`, and `<date>-gate1-map-latency.mov` (or the camera's native extension) under `data/recordings/`. Log phone count, the detection ID, the taped and estimated points, and video latency.

**Pass:** Four phones join and sync, and one clap produces one map pin. Check the clap-to-map target of under 0.6 s using the video. The NPZ timing report measures an approximate server onset-to-broadcast interval.

## Gate 2 · Ten scored positions

Export any prior trials before starting a fresh table with **Clear all trials**. Once recording, do a known-point check in judge mode with **Check a known point** and confirm the error is below 0.5 m. Prefer taped points inside the phone hull for rehearsals. Record any judge-selected outside points as played.

```bash
NAME="$DAY-gate2"
preflight --require-sync
capture_start "$NAME"
# Set the known point, clap, then click Check a known point.
# For each position, set ground truth and sound class and click Start trial.
# Chirp, wait for the listening prompt, then play the sound.
# Confirm the matching detection or log a miss. Repeat for all ten positions.
demo "$BASE/api/eval/export.csv" > "$OUT/$NAME.csv"
demo "$BASE/api/eval/trials" > "$OUT/$NAME-trials.json"
demo "$BASE/api/eval/known-point" > "$OUT/$NAME-known-points.json"
capture_stop
python -m scripts.phone_gate timing "$OUT/$NAME.npz" | tee "$OUT/$NAME-timing.txt"
python -m scripts.recording_summary "$OUT/$NAME.npz"
```

**Artifacts:** `<date>-gate2.npz`, `<date>-gate2.csv`, `<date>-gate2-trials.json`, `<date>-gate2-known-points.json`, and `<date>-gate2-timing.txt` under `data/recordings/`. Log the known-point error, mean, median and p90 localization error, and misses out of ten. Inspect the export for all ten rows before resetting anything. Use the run book's `t_s` annotations when tuning against the recording.

**Pass:** Known point below 0.5 m and ten judged trials exported with errors and misses. Check the real-room mean target of under 0.3 m. If mean error is 0.5 m or greater, replay the capture and review diagnostic plots before sweeping parameters.

## Gate 3 · Ten-minute real drift soak

```bash
NAME="$DAY-gate3-soak"
preflight
capture_start "$NAME"
```

Calibrate and enable auto-chirp every 30 seconds on the dashboard. Once synced, start a ten-minute timer. Leave four real phones joined on venue Wi-Fi and keep the dashboard tab visible. Make taped-point claps between chirps and note the expected detection count. Do not manually recalibrate during the timed interval. Save final preflight before stopping the capture.

```bash
preflight --require-sync | tee "$OUT/$NAME-final-state.txt"
capture_stop
python -m scripts.phone_gate timing "$OUT/$NAME.npz" | tee "$OUT/$NAME-timing.txt"
python -m scripts.recording_summary "$OUT/$NAME.npz"
```

**Artifacts:** `<date>-gate3-soak.npz`, `<date>-gate3-soak-timing.txt`, and `<date>-gate3-soak-final-state.txt` under `data/recordings/`. Log elapsed time, number of successful and attempted solves, worst per-phone residual, phone disconnects/clock jumps, loss percentage per phone, and detections versus claps. Keep the per-solve table to spot drift instead of relying on a final sync badge.

**Pass:** Four real phones stream for ten minutes without manual re-sync, residuals stay below 1 ms, and dropped frames are below 1% on every phone. A disconnect or clock break leaves coverage unknown across that interval. Capture another clean run if it prevents evaluating the target.

For a separate server regression check, run `python scripts/soak.py --minutes 10 --out "$OUT/$DAY-virtual-soak.md"`. The virtual script reports process memory and detection rate in addition to residuals and latency.

## Two mock judged runs

Repeat the ten-position Gate 2 protocol twice with `NAME="$DAY-mock-01"` and `NAME="$DAY-mock-02"`. Use new taped locations, an operator acting as judge, and the actual demo speaker and projector. Export each run before clearing the trial table for the next one. Film multiple claps and record their video latency with the matching event IDs.

**Artifacts:** One NPZ, CSV, trial JSON, known-point JSON, timing report, and latency video per mock under `data/recordings/`. Log each run independently with errors, misses, frame loss, video latency and the projector check.

**Pass:** Both runs have independent ten-row CSVs and matching audio, both known-point checks are under 0.5 m, and the projector view is readable from the back of the room. Record the localization and latency targets for each run.

## Log the sessions and freeze the release

Append one analysis row per recorded gate or mock. Substitute measurements and the actual decision for the sample text.

```bash
python -m scripts.recording_summary "$OUT/$NAME.npz" \
  --append-log analysis/LOG.md --date "$DAY" --what 'Gate 1' \
  --result 'PASS. Add the measured residual, mapped point and video latency here.'
```

Keep the raw audio files in `data/recordings/` and share them with the team separately. Commit the measured log rows with the operator-approved gate outcome.

Run release verification before the gates or during prep with `bash scripts/freeze_release.sh --verify-only`. After all gates and both mock runs have been reviewed, check out the agreed clean release commit on `main`. Test the backup hotspot or travel router by restarting the LAN launcher, rejoining the four phones, calibrating and getting one new map pin. Keep a separate `<date>-backup-network.npz` and log that result. If LAN is unavailable, use the run book's tunnel mode for the backup rehearsal.

```bash
bash scripts/freeze_release.sh v1.0.0
```

Use the team's agreed tag in place of `v1.0.0`. **Pass:** Full Python tests, frontend tests and build pass on the tagged commit, and the backup network check passes. The script creates a local annotated tag. Get explicit approval before pushing the tag.

## Watch these live paths first

| Risk | Where to look | What to check |
| --- | --- | --- |
| Buffer gaps | `PhoneRingBuffer.append`, `Recording.append`, aligned event extraction | Verify the recorded valid mask and frame counters. A missing chunk can exclude a phone from localization. |
| Reference phone dropout | `Session._invalidate_clock`, `SyncEngine.mark_clock_jump` | Verify the remaining phones have valid re-referenced clocks. An epoch change resets pending event groups. |
| Calibration solve | `Session.arm` and `Session.solve` | Check per-phone residuals and confirm the chirp falls after arm. Changes in position or connection state during solve require a fresh arm. |
| Event grouping | `EventGrouper.pop_ready` and `EventLoop.process_once` | Check that three synced phones receive the same onset and buffer audio past the playout window. Check lingering audio or chirp suppression if a pin fails to appear. |
| WebSocket reconnect | `phone_socket` and `frontend/src/phone/main.ts` | Verify the ID and measured position survive, the first returning frame marks the clock jump, and the next chirp restores sync. |
