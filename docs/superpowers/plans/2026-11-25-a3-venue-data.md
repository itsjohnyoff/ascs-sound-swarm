# Agent 3 Venue Data, Classifier Eval & Spike AGC Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the venue audio capture protocol (`analysis/venue_capture.md`), an offline heuristic-vs-trained classifier evaluator (`analysis/classifier_eval.py`), and Gate 0 AGC/gating detection in `analysis/spike.py` so tomorrow's real-phone venue session can decide Gate 0 and pick the live classifier from real venue F1.

**Architecture:** All three deliverables build on the existing `backend.dsp.replay.load_recording_npz` loader, `backend.server.events.find_events_offline`, and `backend.classifier.model.SoundClassifier` without touching `shared/` or `backend/server/`. `classifier_eval.py` handles both simulator `.npz` recordings (ground-truth events in `meta["truth"]["events"]`) and real venue `.npz` recordings (per-file class labels inferred from `meta["label"]` or filename stem, slicing events with or without chirps). `spike.py` adds a lightweight per-phone `agc_gated` check to `PhoneReport` and its Markdown table.

**Tech Stack:** Python 3.13, NumPy, SciPy (already installed), scikit-learn (`models/drone_classifier.joblib`), pytest. No new dependencies.

**Spec:** `CONTRACTS.md` sections 7 and 8; `AGENT_3_harness_classifier.md`; user task specification for Agent 3 (`feat/a3-venue-eval-tooling`).

## Global Constraints

- Edit only these paths: `sim/`, `analysis/`, `backend/classifier/`, `models/`, `tests/sim/`, `tests/classifier/` (plus this plan in `docs/superpowers/plans/`). Never edit `shared/` without Agent 1.
- Never push to, rebase, or modify PR #9's branch (`feat/a3-trained-classifier`). Commit only on `feat/a3-venue-eval-tooling`.
- Keep `models/drone_classifier.joblib` under 10 MB; never download full DroneAudioSet; never commit recordings over 50 MB.
- No new dependencies.
- Branch names must never contain `"codex"`; PR bodies must never contain `"Generated with Claude Code"`.

## Review Focus

- **Venue `.npz` recordings without chirps:** Real 20 s venue clips (`*-venue-claps.npz`, `*-venue-noise.npz`, `*-venue-drone.npz`) may be recorded without running a chirp calibration first. `classifier_eval.py` must slice multi-phone windows directly (resampled to 48 kHz) rather than returning 0 events when `SyncEngine` has no chirps.
- **Label mapping for simulator vs venue files:** Simulator truth uses `"speech"` (or other negative kinds) that map to `"other"` in `CLASSES = ("drone", "clap", "other")`. Venue filenames (`2026-11-26-venue-noise.npz`) must map `"noise"` → `"other"`, `"claps"`/`"clap"` → `"clap"`, `"drone"` → `"drone"`.
- **Missing or broken trained model:** If `models/drone_classifier.joblib` is missing or invalid, `classifier_eval.py` must still run, clearly label the trained column as fell-back (`heuristic`), and never crash.
- **Empty or very short recordings:** Recordings shorter than a single window or with zero detected events must raise a clear `ValueError` or report `0` events cleanly without `ZeroDivisionError` in F1 or timing math.
- **AGC/gating false positives on clean simulated audio:** `spike.py`'s `agc_gated` flag must be `False` on clean simulated recordings (`simulate(...)` with normal noise floor and echoes) and `True` on phones with digital-zero noise gating or > 6 dB post-clap AGC gain ducking.

---

### Task 1: Venue Capture Protocol (`analysis/venue_capture.md`)

**Files:**
- Create: `analysis/venue_capture.md`
- Test: `tests/sim/test_venue_capture_doc.py`

**Interfaces:**
- Consumes: CLI invocations of `analysis.spike` and `analysis.classifier_eval`, plus `analysis/LOG.md` table format.
- Produces: Operator-ready runbook `analysis/venue_capture.md` naming exact files (`data/recordings/<date>-venue-claps.npz`, `data/recordings/<date>-venue-noise.npz`, `data/recordings/<date>-venue-drone.npz`), durations, speaker placement, and exact `analysis/LOG.md` lines.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

DOC = Path("analysis/venue_capture.md")

def test_venue_capture_protocol_has_required_sections():
    text = DOC.read_text(encoding="utf-8")
    for token in (
        "spike-01.npz",
        "-venue-claps.npz",
        "-venue-noise.npz",
        "-venue-drone.npz",
        "20 s",
        "python -m analysis.spike",
        "python -m analysis.classifier_eval",
        "analysis/LOG.md",
        "Gate 0",
        "macro F1",
    ):
        assert token in text, f"missing required token: {token}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/sim/test_venue_capture_doc.py -v`
Expected: FAIL with `FileNotFoundError` for `analysis/venue_capture.md`.

- [ ] **Step 3: Write `analysis/venue_capture.md`**

Include concise, exact sections:
1. **Pre-flight & Gate 0 Spike (`data/recordings/spike-01.npz`)**: phone placement, browser mic settings (`echoCancellation: false`, `noiseSuppression: false`, `autoGainControl: false`), 3 chirps + 5 claps at taped `(x_m, y_m)` points, command `python -m analysis.spike data/recordings/spike-01.npz`, Gate 0 pass thresholds (residual < 1 ms, 0 gaps, clipped < 1%, AGC/gate OK).
2. **Venue Classifier Clips**: exact filenames `data/recordings/<date>-venue-claps.npz`, `data/recordings/<date>-venue-noise.npz`, `data/recordings/<date>-venue-drone.npz` (each ≥ 20 s; 25 s recommended with 2 s buffer), exact speaker placement (actual demo speaker, 2+ positions inside the phone polygon at 1.5–4 m from phones, realistic demo volume without clipping), clap cadence (~1 clap every 1.5–2 s for 10+ claps), room noise (normal venue chatter/HVAC with no claps or drone).
3. **Evaluation & Live-Model Decision**: exact command `python -m analysis.classifier_eval data/recordings/<date>-venue-claps.npz data/recordings/<date>-venue-noise.npz data/recordings/<date>-venue-drone.npz`, target macro F1 > 0.9, rule for picking live backend based on venue F1 only (and how to drop `models/drone_classifier.joblib` to force heuristic if heuristic wins).
4. **Exact `analysis/LOG.md` Lines**: copy-pasteable template rows for both `spike-01` (Gate 0) and the venue F1 comparison.
5. **Size & Repo Hygiene**: never commit `.npz` recordings > 50 MB.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/sim/test_venue_capture_doc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/venue_capture.md tests/sim/test_venue_capture_doc.py
git commit -m "docs(analysis): add venue audio capture protocol for Gate 0 and classifier eval"
```

---

### Task 2: Offline Classifier Evaluator (`analysis/classifier_eval.py`)

**Files:**
- Create: `analysis/classifier_eval.py`
- Test: `tests/classifier/test_classifier_eval.py`

**Interfaces:**
- Consumes: `backend.dsp.replay.load_recording_npz`, `backend.server.events.find_events_offline`, `backend.server.events.TARGET_FS`, `backend.classifier.model.SoundClassifier`, `backend.classifier.model.CLASSES`, `backend.classifier.model.DEFAULT_MODEL_PATH`.
- Produces:
  - `LabeledEvent(source: str, label: str, aligned: dict[str, np.ndarray], fs: int)`
  - `BackendMetrics(name: str, backend_name: str, confusion: list[list[int]], per_class_f1: dict[str, float], macro_f1: float, median_ms: float, p90_ms: float, n_events: int)`
  - `extract_labeled_events(paths: Sequence[str | Path]) -> list[LabeledEvent]`
  - `evaluate_backend(events: Sequence[LabeledEvent], classifier: SoundClassifier, name: str) -> BackendMetrics`
  - `evaluate_files(paths: Sequence[str | Path], model_path: str | Path | None = DEFAULT_MODEL_PATH) -> tuple[BackendMetrics, BackendMetrics]`
  - `format_eval_report(heuristic: BackendMetrics, trained: BackendMetrics) -> str`
  - CLI `python -m analysis.classifier_eval data/recordings/<files>.npz [--model PATH] [--json]`

- [ ] **Step 1: Write the failing tests in `tests/classifier/test_classifier_eval.py`**

Tests to write:
1. `test_eval_on_simulator_judge_recording(tmp_path)`: saves `judge_scenario(seed=0)` to `tmp_path / "sim-judge.npz"`, calls `evaluate_files([path])`, asserts `heuristic.n_events >= 8`, `trained.n_events == heuristic.n_events`, `0.0 <= heuristic.macro_f1 <= 1.0`, `0.0 <= trained.macro_f1 <= 1.0`, `heuristic.median_ms > 0`, `trained.median_ms > 0`, confusion matrix shape is `3x3` summing to `n_events`, and `format_eval_report(heuristic, trained)` contains `"macro F1"`, `"ms/event"`, `"drone"`, `"clap"`, `"other"`.
2. `test_eval_on_uncalibrated_venue_style_npz_files(tmp_path)`: simulates 3 standalone recordings WITHOUT chirps (`2026-11-26-venue-claps.npz`, `2026-11-26-venue-noise.npz`, `2026-11-26-venue-drone.npz`, stripping `truth` from `meta` so they match real venue `.npz` files), runs `evaluate_files([...])`, verifies all 3 classes (`clap`, `other`, `drone`) have nonzero ground-truth counts in the confusion matrix and both backends return valid F1 and timing metrics.
3. `test_eval_missing_model_falls_back_gracefully(tmp_path)`: calls `evaluate_files([path], model_path=tmp_path / "missing.joblib")`, verifies `trained.backend_name == "heuristic"` without crashing.
4. `test_eval_cli_json_and_markdown(tmp_path, capsys)`: invokes `main([str(path)])` and `main([str(path), "--json"])`, verifies exit code 0 and valid JSON output with `heuristic` and `trained` keys.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/classifier/test_classifier_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.classifier_eval'`.

- [ ] **Step 3: Implement `analysis/classifier_eval.py`**

Key implementation details:
- Label normalization: `"drone"` → `"drone"`, `"clap"` / `"claps"` → `"clap"`, anything else (`"noise"`, `"other"`, `"speech"`, `"knock"`, etc.) → `"other"`.
- Event slicing in `extract_labeled_events`:
  - If `meta.get("truth", {}).get("events")` exists (simulator recording): run `find_events_offline(slices, phones, room, min_phones=2)` and match each detection to the nearest non-chirp truth event within `0.5 s` (using `SyncEngine` + `sim.room.true_time`), returning `LabeledEvent(path.name, normalize_label(target["kind"]), det.aligned, TARGET_FS)`.
  - Else (real venue `.npz` recording): infer label from `meta.get("label")` or filename stem (`clap` / `drone` / `noise` / `other`). Resample each phone's valid PCM to `TARGET_FS` (48 kHz) via `scipy.signal.resample_poly` when `s.sample_rate != TARGET_FS`:
    - If label is `"clap"`: find transient peaks on the loudest phone (RMS in 10 ms hop exceeding `max(6 * median_rms, 0.02)`, min separation `0.35 s`) and slice `[-0.015 s, +0.065 s]` (80 ms) aligned across all phones.
    - If label is `"drone"` or `"other"`: slice non-overlapping `0.75 s` windows (trimming `0.25 s` at edges; for `"drone"`, keep windows whose max-phone RMS ≥ 20% of recording median RMS so leading/trailing silence isn't mislabeled).
- Confusion matrix & macro F1: pure NumPy/Python over `CLASSES = ("drone", "clap", "other")` (F1 = `2*tp / (2*tp + fp + fn)` or `0.0` if denominator is 0; macro F1 averages over classes present in ground truth).
- Timing: measure each `classifier.classify_swarm(ev.aligned, fs=ev.fs)` with `time.perf_counter()`, reporting median and p90 ms/event.
- `format_eval_report`: Markdown tables for Heuristic vs Trained (confusion matrices, macro F1, median/p90 ms/event) + recommendation line based on macro F1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/classifier/test_classifier_eval.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add analysis/classifier_eval.py tests/classifier/test_classifier_eval.py
git commit -m "feat(analysis): add offline classifier evaluator for simulator and venue recordings"
```

---

### Task 3: Extend `analysis/spike.py` with Gate 0 AGC/Gating Flag

**Files:**
- Modify: `analysis/spike.py:27-125`
- Test: `tests/sim/test_spike_analysis.py`

**Interfaces:**
- Consumes: `PhoneAudioSlice` (`samples_f32`, `sample_rate`, `valid_mask`).
- Produces:
  - `_detect_agc_or_gating(samples: np.ndarray, fs: int, valid_mask: np.ndarray | None = None) -> tuple[bool, str]`
  - `PhoneReport.agc_gated: bool = False` and `PhoneReport.agc_reason: str = ""`
  - Updated `format_report(r: SpikeReport)` adding `AGC/gate` column (`OK` or `FLAG (<reason>)`).

- [ ] **Step 1: Write the failing tests in `tests/sim/test_spike_analysis.py`**

Add tests:
1. Assert in `test_spike_report_on_simulated_session` that `all(not p.agc_gated for p in report.phones.values())` and `"AGC/gate"` appears in `format_report(report)`.
2. `test_spike_flags_noise_gated_phone(tmp_path)`: simulate a session and zero out quiet frames (`abs(x) < 0.005`) on `p3` to simulate aggressive browser noise gating; assert `report.phones["p3"].agc_gated is True` and `report.phones["p1"].agc_gated is False`.
3. `test_spike_flags_agc_pumping_after_claps(tmp_path)`: simulate a session with 3 claps and apply a simulated AGC gain dip (multiply samples 100–450 ms after each clap peak by `0.25`) on `p4`; assert `report.phones["p4"].agc_gated is True` and `report.phones["p1"].agc_gated is False`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/sim/test_spike_analysis.py -v`
Expected: FAIL with `AttributeError: 'PhoneReport' object has no attribute 'agc_gated'`.

- [ ] **Step 3: Implement `_detect_agc_or_gating` and wire into `PhoneReport` / `format_report` in `analysis/spike.py`**

- Compute 20 ms frame RMS over valid samples (`valid_mask` True):
  1. **Gated silence**: if signal has active audio (`peak >= 0.02`) and > 10% of valid 20 ms frames have `rms < 1e-5` (hard digital mute between sounds), return `(True, "noise-gated")`.
  2. **AGC gain ducking**: find loud transient peaks (`frame_rms > max(8 * median_rms, 0.03)` separated by ≥ 1.0 s); for each peak with valid pre-window `[-450 ms, -150 ms]` and post-window `[+150 ms, +450 ms]`, compare median post-window RMS to median pre-window RMS. If at least 2 peaks show `post_rms < 0.45 * pre_rms` (a > 7 dB gain dip right after a clap), return `(True, "agc-ducking")`.
  3. Otherwise return `(False, "")`.
- Add `agc_gated: bool = False` and `agc_reason: str = ""` to `PhoneReport` and show `AGC/gate` column in `format_report()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/sim/test_spike_analysis.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add analysis/spike.py tests/sim/test_spike_analysis.py
git commit -m "feat(analysis): detect AGC pumping and noise gating in spike Gate 0 report"
```
