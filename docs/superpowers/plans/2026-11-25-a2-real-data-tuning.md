# Agent 2 Real-Data Tuning Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose all tunable DSP constants with unchanged defaults, add a bounded grid-search tuning CLI (`backend/dsp/tune.py`), and add single-event diagnostic plots (`backend/dsp/diagnostics.py` -> `analysis/plots/`), all verified on simulator recordings so real recordings can be tuned within minutes of capture.

**Architecture:** Three independent deliverables built with TDD: (1) wire existing hardcoded constants in `sync.py`, `chirp_detect.py`, `localizer.py`, `pipeline.py`, and `trigger.py` as optional keyword parameters whose defaults match current values bit-for-bit; (2) build `backend/dsp/tune.py` to sweep up to 200 parameter combinations against judge CSV or recording-meta ground truth and write a sorted Markdown table to `analysis/tuning-<name>.md`; (3) build `backend/dsp/diagnostics.py` to render chirp-envelope, per-pair GCC-PHAT, and triangle-residual plots into `analysis/plots/` for any single event.

**Tech Stack:** Python 3, NumPy, SciPy, Matplotlib (already installed; no new dependencies), pytest.

**Spec:** User task prompt (Agent 2 Phase 3 tuning harness, CONTRACTS.md §§2, 4, 7).

## Tunable Constants Map (Source-Verified)

| Parameter | File:Line | Current Value | Exposed On |
| --- | --- | --- | --- |
| Chirp bandpass low edge | `backend/dsp/sync.py:98` | `1200.0` Hz | `_compute_chirp_envelope`, `detect_all_chirps`, `detect_chirp_arrivals`, `detect_chirps`, `SyncEngine` |
| Chirp bandpass high edge | `backend/dsp/sync.py:97` | `8500.0` Hz (capped at `0.48 * fs`) | `_compute_chirp_envelope`, `detect_all_chirps`, `detect_chirp_arrivals`, `detect_chirps`, `SyncEngine` |
| Chirp lookback window | `backend/dsp/sync.py:145` | `0.025` s (25 ms; spec mentions 20 ms) | `_pick_first_arrival_with_ncc`, `_pick_first_arrival`, `detect_all_chirps`, `detect_chirp_arrivals`, `detect_chirps`, `SyncEngine` |
| Chirp direct-path ratio | `backend/dsp/sync.py:148` | `0.55` (spec mentions 0.65) | `_pick_first_arrival_with_ncc`, `_pick_first_arrival`, `detect_all_chirps`, `detect_chirp_arrivals`, `detect_chirps`, `SyncEngine` |
| GCC-PHAT low band edge | `backend/dsp/gcc_phat.py:24` | `300.0` Hz | `compute_gcc_phat` (existing), `AcousticLocalizer2D`, `LocalizationPipeline` |
| GCC-PHAT high band edge | `backend/dsp/gcc_phat.py:25` | `6000.0` Hz | `compute_gcc_phat` (existing), `AcousticLocalizer2D`, `LocalizationPipeline` |
| GCC-PHAT beta | `backend/dsp/gcc_phat.py:26` | `0.8` | `compute_gcc_phat` (existing), `AcousticLocalizer2D`, `LocalizationPipeline` |
| Triangle cycle-consistency threshold | `backend/dsp/localizer.py:239` | `0.0015` s (1.5 ms) | `AcousticLocalizer2D`, `LocalizationPipeline` |
| Leave-one-out residual threshold | `backend/dsp/localizer.py:306` | `0.0005` s (0.5 ms) | `AcousticLocalizer2D`, `LocalizationPipeline` |
| Sustained Hann frame length | `backend/dsp/pipeline.py:68`, `backend/dsp/trigger.py:105` | `0.250` s (3 frames, 50% hop) | `LocalizationPipeline`, `AdaptiveEventTrigger` |
| Transient crop pre/post window | `backend/dsp/trigger.py:92-93` | `0.015` s / `0.065` s | `AdaptiveEventTrigger`, `tune.py` extraction |
| Onset SNR threshold | `backend/dsp/trigger.py:132` | `10.0` dB | `OnsetDetector` (already exposed) |
| Onset refractory period | `backend/dsp/trigger.py:133` | `0.15` s (150 ms) | `OnsetDetector` (already exposed) |

## Global Constraints

- Edit only: `backend/dsp/`, `tests/dsp/`, new files under `analysis/`, one-line appends to `analysis/LOG.md`, and this plan doc. Never edit `shared/`, `backend/server/`, or `sim/`.
- Pure NumPy, SciPy, and already-installed Matplotlib only. No new dependencies.
- Speed of sound MUST come from `shared.protocol.speed_of_sound(room.temperature_c)`. Never hardcode `343.0`.
- Default behavior MUST NOT change: all existing tests pass and `python -m sim.benchmark` mean error stays within `0.02 m` of `0.095 m`.
- Grid search in `backend/dsp/tune.py` is hard-capped at `<= 200` combinations and prints wall-clock time.
- Do not tune defaults against simulator data.

## Review Focus

1. **Missing/unmatched events in `tune.py`**: If a parameter combo misses all events, mean/median/p90 must format as `-` and sort after combos with detections (never crash on `np.mean([])`).
2. **Judge CSV vs meta ground truth formats**: `tune.py` must accept both `judge_eval.py` CSV (`gt_x_m,gt_y_m,...`), minimal CSV (`x_m,y_m`), `meta["clap_points"]`, and simulator `meta["truth"]["events"]`.
3. **Grid size cap enforcement**: Passing a grid with `> 200` combinations must raise `ValueError` (never silently run thousands of combos).
4. **Single-event diagnostics with 2, 3, or 4+ phones**: Triangle residuals require `>= 3` phones; with 2 phones, plot an informative empty/N/A panel rather than crashing.
5. **No default behavior drift**: Passing explicit default values to `SyncEngine`, `detect_all_chirps`, `AcousticLocalizer2D`, and `LocalizationPipeline` produces identical outputs to calling with no arguments.

---

### Task 1: Expose Tunable DSP Constants with Unchanged Defaults

**Files:**
- Modify: `backend/dsp/sync.py:79-184,201-305,320-326,401-435,683-686`
- Modify: `backend/dsp/chirp_detect.py:13-34`
- Modify: `backend/dsp/localizer.py:53-59,142-146,238-242,306-307`
- Modify: `backend/dsp/trigger.py:19-24,89-118`
- Modify: `backend/dsp/pipeline.py:32-39,65-79`
- Test: `tests/dsp/test_tunable_params.py`

**Interfaces:**
- Produces:
  - `detect_all_chirps(..., lookback_s: float = 0.025, direct_path_ratio: float = 0.55, f_low: float = 1200.0, f_high: float = 8500.0) -> list[ChirpArrival]`
  - `detect_chirps(..., lookback_s: float = 0.025, direct_path_ratio: float = 0.55, f_low: float = 1200.0, f_high: float = 8500.0) -> list[Arrival]`
  - `SyncEngine(*, chirp_lookback_s: float = 0.025, chirp_direct_path_ratio: float = 0.55, chirp_f_low: float = 1200.0, chirp_f_high: float = 8500.0)`
  - `AcousticLocalizer2D(grid_spacing_m: float = 0.15, *, f_min: float = 300.0, f_max: float = 6000.0, phat_beta: float = 0.8, cycle_threshold_s: float = 0.0015, loo_residual_threshold_s: float = 0.0005)`
  - `LocalizationPipeline(grid_spacing_m: float = 0.15, *, f_min: float = 300.0, f_max: float = 6000.0, phat_beta: float = 0.8, cycle_threshold_s: float = 0.0015, loo_residual_threshold_s: float = 0.0005, sustained_frame_s: float = 0.250, sustained_max_frames: int = 3, transient_pre_s: float = 0.015, transient_post_s: float = 0.065)`

- [ ] **Step 1: Write failing tests in `tests/dsp/test_tunable_params.py`**

Tests to write:
1. `test_default_params_match_baseline_sync_and_localize`: construct default vs explicit-default `SyncEngine` and `LocalizationPipeline`, run on a synthetic 4-phone clap + chirp signal, assert identical arrival `sample_index` (exact equality) and identical `(x_m, y_m, residual_ms)`.
2. `test_chirp_direct_path_ratio_and_lookback_affect_echo_picking`: construct a signal with a direct-path chirp at 0.45x amplitude followed 12 ms later by a 1.0x echo; assert `direct_path_ratio=0.55` picks the echo while `direct_path_ratio=0.40, lookback_s=0.020` picks the earlier direct path.
3. `test_localizer_band_beta_and_cycle_threshold_are_wired`: verify `LocalizationPipeline(f_min=500.0, f_max=4000.0, phat_beta=0.6, cycle_threshold_s=0.0005, loo_residual_threshold_s=0.0003, sustained_frame_s=0.200)` stores and forwards parameters into `AcousticLocalizer2D` and `compute_gcc_phat`, and that a narrow out-of-band filter changes the result.

- [ ] **Step 2: Run `pytest tests/dsp/test_tunable_params.py -v`**
Expected: FAIL (`TypeError: unexpected keyword argument`).

- [ ] **Step 3: Implement parameter wiring in `sync.py`, `chirp_detect.py`, `localizer.py`, `trigger.py`, `pipeline.py`**
- Thread `f_low`, `f_high`, `lookback_s`, `direct_path_ratio` through `_compute_chirp_envelope`, `_pick_first_arrival_with_ncc`, `_pick_first_arrival`, `detect_all_chirps`, `detect_chirp_arrivals`, `detect_chirps`, and `SyncEngine.__init__`.
- Thread `f_min`, `f_max`, `phat_beta`, `cycle_threshold_s`, `loo_residual_threshold_s` into `AcousticLocalizer2D` and `LocalizationPipeline`.
- Thread `sustained_frame_s`, `sustained_max_frames`, `transient_pre_s`, `transient_post_s` into `AdaptiveEventTrigger` and `LocalizationPipeline`.

- [ ] **Step 4: Run `pytest tests/dsp -v`**
Expected: PASS (all existing + new tests pass).

---

### Task 2: Grid-Search Tuning Harness (`backend/dsp/tune.py`)

**Files:**
- Create: `backend/dsp/tune.py`
- Test: `tests/dsp/test_tune.py`

**Interfaces:**
- Consumes: Task 1's `SyncEngine`, `LocalizationPipeline`, `OnsetDetector`, `load_recording_npz`, `EventGrouper`, `decide_kind`, `still_sustained`, `sustained_hold_until`, `max_spread_s`, `chirp_suppression`, `TARGET_FS`, `speed_of_sound`.
- Produces:
  - `GroundTruthEvent(x_m: float, y_m: float, t_s: float | None = None, kind: str | None = None, label: str = "")`
  - `TuneParams(f_min: float = 300.0, f_max: float = 6000.0, phat_beta: float = 0.8, cycle_threshold_s: float = 0.0015, event_window_post_s: float = 0.065, chirp_direct_path_ratio: float = 0.55, chirp_lookback_s: float = 0.025, onset_snr_db: float = 10.0, onset_refractory_s: float = 0.15)`
  - `TuneRow(params: TuneParams, mean_error_m: float | None, median_error_m: float | None, p90_error_m: float | None, missed: int, total: int, runtime_ms: float)`
  - `load_ground_truth(meta: dict, truth_csv: str | Path | None = None) -> list[GroundTruthEvent]`
  - `sweep_recording(npz_path: str | Path, *, truth_csv: str | Path | None = None, grid: list[TuneParams] | None = None, max_combos: int = 200) -> tuple[list[TuneRow], float]`
  - `format_tuning_markdown(rows: list[TuneRow], *, recording_name: str, elapsed_s: float) -> str`
  - CLI: `python -m backend.dsp.tune <recording.npz> [--truth truth.csv] [--out analysis] [--max-combos 200]` -> prints wall-clock time & Markdown table, saves `analysis/tuning-<stem>.md`.

- [ ] **Step 1: Write failing tests in `tests/dsp/test_tune.py`**

Tests to write:
1. `test_load_ground_truth_csv_and_meta`: verify loading from a `judge_eval`-format CSV (`gt_x_m,gt_y_m,...`), a minimal CSV (`x_m,y_m`), `meta["clap_points"]`, and simulator `meta["truth"]["events"]`.
2. `test_sweep_caps_grid_at_200`: verify `sweep_recording(..., grid=[TuneParams()]*201)` raises `ValueError`.
3. `test_tune_cli_on_sim_recording`: generate a simulator recording via `sim.room.judge_scenario(seed=0)` saved to `tmp_path/sim_test.npz`, run `main([str(npz_path), "--out", str(tmp_path)])` with a small 2-combo grid (or fast flag), verify `tmp_path/tuning-sim_test.md` is created, contains sorted table columns (`mean_m`, `median_m`, `p90_m`, `missed`), prints wall-clock time, and has default-row mean error `< 0.20 m` with `0/10` missed.
4. `test_format_markdown_handles_all_missed`: verify a `TuneRow` with `mean_error_m=None, missed=10, total=10` formats cleanly as `-` and sorts after rows with finite mean error.

- [ ] **Step 2: Run `pytest tests/dsp/test_tune.py -v`**
Expected: FAIL (`ModuleNotFoundError: No module named 'backend.dsp.tune'`).

- [ ] **Step 3: Implement `backend/dsp/tune.py`**
- Reuse `SyncEngine` per unique `(chirp_direct_path_ratio, chirp_lookback_s)` and onset detection per unique `(chirp_direct_path_ratio, chirp_lookback_s, onset_snr_db, onset_refractory_s)` so sweeping localization parameters (`f_min`, `f_max`, `phat_beta`, `cycle_threshold_s`, `event_window_post_s`) is fast.
- Default grid: compact curated sweep `<= 200` combinations (e.g., 3 bands × 3 `phat_beta` × 2 `cycle_threshold_s` × 2 `event_window_post_s` × 2 `chirp_direct_path_ratio` = 72 combos, always including exact defaults).
- Match detections to ground truth via `true_time` when `meta["truth"]["phones"]` exists, else chronological order (or timestamp proximity if CSV provides matching timestamps).
- Sort rows by `(missed_all, mean_error_m, missed)` ascending and save to `analysis/tuning-<stem>.md`.

- [ ] **Step 4: Run `pytest tests/dsp/test_tune.py -v`**
Expected: PASS.

---

### Task 3: Single-Event Diagnostic Plots (`backend/dsp/diagnostics.py` -> `analysis/plots/`)

**Files:**
- Create: `backend/dsp/diagnostics.py`
- Test: `tests/dsp/test_diagnostics.py`

**Interfaces:**
- Consumes: Task 1's `_compute_chirp_envelope`, `detect_all_chirps`, `SyncEngine`, `compute_gcc_phat`, `LocalizationPipeline`, `load_recording_npz`, `find_events_offline`, `speed_of_sound`.
- Produces:
  - `EventDiagnostics(event_index: int, t0_s: float, kind: str, est_xy: tuple[float, float] | None, pair_tdoas_ms: dict[str, float], triangle_residuals_ms: dict[str, float], plot_paths: dict[str, Path])`
  - `generate_event_diagnostics(npz_path: str | Path, *, event_index: int = 0, out_dir: str | Path = "analysis/plots", chirp_direct_path_ratio: float = 0.55, chirp_lookback_s: float = 0.025, f_min: float = 300.0, f_max: float = 6000.0, phat_beta: float = 0.8, cycle_threshold_s: float = 0.0015) -> EventDiagnostics`
  - CLI: `python -m backend.dsp.diagnostics <recording.npz> [--event 0] [--out analysis/plots]`

- [ ] **Step 1: Write failing tests in `tests/dsp/test_diagnostics.py`**

Tests to write:
1. `test_diagnostics_generates_all_three_plots_on_sim_recording`: generate `judge_scenario(seed=0)` into `tmp_path/diag_sim.npz`, run `generate_event_diagnostics(npz_path, event_index=0, out_dir=tmp_path / "plots")`, assert `plot_paths` has keys `"chirp_envelope"`, `"gcc_pairs"`, `"triangle_residuals"`, all 3 PNG files exist with size `> 1000` bytes, `len(pair_tdoas_ms) == 6` (4 phones), and `len(triangle_residuals_ms) == 4` (C(4,3)=4 triangles) with max residual `< 1.0 ms`.
2. `test_diagnostics_two_phone_recording_does_not_crash`: run on a 2-phone simulated recording (`judge_scenario(n_phones=2)` or `simulate` with 2 phones), assert `triangle_residuals_ms == {}` and all 3 PNGs are still created cleanly.
3. `test_diagnostics_invalid_event_index_raises`: assert `IndexError` when `event_index` is out of range.

- [ ] **Step 2: Run `pytest tests/dsp/test_diagnostics.py -v`**
Expected: FAIL (`ModuleNotFoundError: No module named 'backend.dsp.diagnostics'`).

- [ ] **Step 3: Implement `backend/dsp/diagnostics.py`**
- Use `matplotlib.use("Agg")` headless backend.
- Plot 1 (`<stem>-event<idx>-chirp-envelope.png`): per-phone matched-filter Hilbert envelope in `±30 ms` around first chirp peak, showing detected direct-path peak, global peak, lookback region, and `direct_path_ratio * peak` line.
- Plot 2 (`<stem>-event<idx>-gcc-pairs.png`): grid of per-pair normalized GCC-PHAT curves (`ms` vs correlation), marking measured peak TDOA, predicted TDOA from localized `(x, y)` (computed with `speed_of_sound(room.temperature_c)`), and PSR weight.
- Plot 3 (`<stem>-event<idx>-triangle-residuals.png`): bar chart of `|tau_ij + tau_jk - tau_ik|` in ms for every phone triple `(i, j, k)` with dashed reference lines at `0.5 ms` and `cycle_threshold_s * 1000 ms` (or an informative "Requires >= 3 phones" panel if fewer than 3 phones).

- [ ] **Step 4: Run `pytest tests/dsp/test_diagnostics.py -v`**
Expected: PASS.
