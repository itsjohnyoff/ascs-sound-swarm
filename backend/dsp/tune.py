"""Bounded grid-search tuning harness for DSP parameters on recorded .npz sessions.

CLI:
    python -m backend.dsp.tune <recording.npz> [--truth truth.csv] [--out analysis] [--max-combos 200]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from backend.dsp.pipeline import LocalizationPipeline, OnsetDetector, SyncEngine
from backend.dsp.replay import RecordedSession, load_recording_npz
from backend.server.events import (
    FRAME,
    SUSTAIN_MAX_UPDATES,
    TARGET_FS,
    EventGrouper,
    chirp_suppression,
    decide_kind,
    event_window,
    max_spread_s,
    still_sustained,
    sustained_hold_until,
)
from shared.protocol import (
    EVENT_WINDOW_POST_S,
    EVENT_WINDOW_PRE_S,
    PLAYOUT_SLACK_S,
    SUSTAINED_WINDOW_S,
    EventKind,
)

MAX_COMBOS = 200
MATCH_WINDOW_S = 0.5


@dataclass(frozen=True, slots=True)
class GroundTruthEvent:
    x_m: float
    y_m: float
    t_s: float | None = None
    kind: str | None = None
    label: str = ""


@dataclass(frozen=True, slots=True)
class TuneParams:
    f_min: float = 300.0
    f_max: float = 6000.0
    phat_beta: float = 0.8
    cycle_threshold_s: float = 0.0015
    event_window_post_s: float = EVENT_WINDOW_POST_S
    chirp_direct_path_ratio: float = 0.55
    chirp_lookback_s: float = 0.025
    onset_snr_db: float = 10.0
    onset_refractory_s: float = 0.15


@dataclass(slots=True)
class TuneRow:
    params: TuneParams
    mean_error_m: float | None
    median_error_m: float | None
    p90_error_m: float | None
    missed: int
    total: int
    runtime_ms: float


def default_grid() -> list[TuneParams]:
    """Compact 72-combo grid covering localization and chirp direct-path parameters."""
    bands = ((300.0, 6000.0), (500.0, 4000.0), (200.0, 8000.0))
    betas = (0.6, 0.8, 1.0)
    cycles = (0.0015, 0.0025)
    posts = (0.050, 0.065)
    ratios = (0.55, 0.65)
    return [
        TuneParams(
            f_min=f_min,
            f_max=f_max,
            phat_beta=beta,
            cycle_threshold_s=cycle,
            event_window_post_s=post,
            chirp_direct_path_ratio=ratio,
        )
        for (f_min, f_max), beta, cycle, post, ratio in product(bands, betas, cycles, posts, ratios)
    ]


def _opt_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _parse_point_list(raw: list[Any]) -> list[GroundTruthEvent]:
    events: list[GroundTruthEvent] = []
    for item in raw:
        if isinstance(item, dict):
            x = item.get("x_m", item.get("x"))
            y = item.get("y_m", item.get("y"))
            if x is None or y is None:
                continue
            t_raw = item.get("t_s", item.get("timestamp_s"))
            kind_raw = item.get("kind", item.get("expected_class"))
            events.append(
                GroundTruthEvent(
                    x_m=float(x),
                    y_m=float(y),
                    t_s=_opt_float(t_raw),
                    kind=str(kind_raw) if kind_raw else None,
                    label=str(item.get("label", "")),
                )
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            events.append(GroundTruthEvent(x_m=float(item[0]), y_m=float(item[1])))
    return events


def load_ground_truth(meta: dict[str, Any], truth_csv: str | Path | None = None) -> list[GroundTruthEvent]:
    """Load ground-truth events from a CSV file or recording metadata."""
    if truth_csv is not None:
        p = Path(truth_csv)
        with p.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or ())
            if {"gt_x_m", "gt_y_m"} <= fields:
                xk, yk = "gt_x_m", "gt_y_m"
            elif {"x_m", "y_m"} <= fields:
                xk, yk = "x_m", "y_m"
            else:
                raise ValueError(f"Truth CSV '{p}' must contain (gt_x_m, gt_y_m) or (x_m, y_m) columns")

            events: list[GroundTruthEvent] = []
            for row in reader:
                if not row.get(xk) or not row.get(yk):
                    continue
                t_raw = row.get("t_s") if "t_s" in fields else row.get("timestamp_s")
                kind_raw = row.get("kind") if "kind" in fields else row.get("expected_class")
                events.append(
                    GroundTruthEvent(
                        x_m=float(row[xk]),
                        y_m=float(row[yk]),
                        t_s=_opt_float(t_raw),
                        kind=str(kind_raw) if kind_raw else None,
                        label=str(row.get("label") or ""),
                    )
                )
            return events

    for key in ("clap_points", "points"):
        raw = meta.get(key)
        if isinstance(raw, list) and raw:
            return _parse_point_list(raw)

    truth = meta.get("truth")
    if isinstance(truth, dict) and isinstance(truth.get("events"), list):
        return [
            GroundTruthEvent(
                x_m=float(e["x_m"]),
                y_m=float(e["y_m"]),
                t_s=_opt_float(e.get("t_s")),
                kind=str(e["kind"]) if e.get("kind") else None,
                label=str(e.get("label", "")),
            )
            for e in truth["events"]
            if isinstance(e, dict) and e.get("kind") != "chirp"
        ]
    return []


def _find_grouped_events(
    session: RecordedSession,
    engine: SyncEngine,
    *,
    onset_snr_db: float,
    onset_refractory_s: float,
    min_phones: int,
) -> list[tuple[float, EventKind]]:
    suppress = chirp_suppression(engine)
    onsets: list[tuple[float, str]] = []
    for pid, slc in session.slices.items():
        if not engine.is_synced(pid):
            continue
        det = OnsetDetector(snr_db_threshold=onset_snr_db, refractory_s=onset_refractory_s)
        for k in range(0, len(slc.samples_f32), FRAME):
            mask = None if slc.valid_mask is None else slc.valid_mask[k : k + FRAME]
            for n in det.push(slc.samples_f32[k : k + FRAME], slc.start_sample_index + k, slc.sample_rate, mask):
                if any(a <= n <= b for a, b in suppress.get(pid, ())):
                    continue
                t = engine.to_global(pid, n)
                if t is not None:
                    onsets.append((t, pid))

    ends = {
        pid: engine.to_global(pid, slc.start_sample_index + len(slc.samples_f32) - 1)
        for pid, slc in session.slices.items()
    }
    valid_ends = [t for t in ends.values() if t is not None]
    if not valid_ends:
        return []
    data_end = min(valid_ends)

    grouper = EventGrouper(max_spread_s(session.room), min_phones)
    grouped: list[tuple[float, EventKind]] = []

    def drain(horizon: float) -> None:
        for ev in grouper.pop_ready(lambda _pid: horizon, slack_s=PLAYOUT_SLACK_S):
            kind = decide_kind(engine, session.slices, ev.t0)
            if kind is None:
                continue
            grouped.append((ev.t0, kind))
            if kind == "sustained":
                seq = 1
                while (
                    seq < SUSTAIN_MAX_UPDATES
                    and event_window(ev.t0, kind, seq)[1] <= data_end
                    and still_sustained(engine, session.slices, ev.t0, seq)
                ):
                    seq += 1
                grouper.hold(sustained_hold_until(ev.t0, seq - 1))

    for t, pid in sorted(onsets):
        drain(t)
        grouper.add(pid, t)
    drain(math.inf)
    return grouped


def _extract_windows(
    session: RecordedSession,
    engine: SyncEngine,
    grouped: list[tuple[float, EventKind]],
    *,
    event_window_post_s: float,
    min_phones: int,
) -> list[tuple[float, EventKind, dict[str, np.ndarray]]]:
    out: list[tuple[float, EventKind, dict[str, np.ndarray]]] = []
    for t0, kind in grouped:
        if kind == "transient":
            lo, hi = t0 - EVENT_WINDOW_PRE_S, t0 + event_window_post_s
        else:
            lo, hi = t0, t0 + SUSTAINED_WINDOW_S
        aligned = engine.extract_aligned_window(session.slices, lo, hi, target_fs=TARGET_FS)
        clean = {pid: x for pid, x in aligned.items() if float(np.max(np.abs(x), initial=0.0)) < 0.99}
        selected = clean if len(clean) >= min_phones else aligned
        if len(selected) >= min_phones:
            out.append((t0, kind, selected))
    return out


def _sim_true_time(truth_phone: dict[str, Any], sample_index: float) -> float:
    k = sample_index - truth_phone["start_index"]
    return float(truth_phone["start_s"] + k / (truth_phone["fs"] * (1.0 + truth_phone["drift_ppm"] * 1e-6)))


def _match_errors(
    detections: list[tuple[float, float, float]],  # (t0, x_m, y_m)
    ground_truth: list[GroundTruthEvent],
    meta: dict[str, Any],
    engine: SyncEngine,
) -> list[float]:
    if not detections or not ground_truth:
        return []

    truth_phones = meta.get("truth", {}).get("phones") if isinstance(meta.get("truth"), dict) else None
    ref = engine.ref_phone_id
    has_gt_times = all(gt.t_s is not None for gt in ground_truth)

    det_times: list[float] | None = None
    if isinstance(truth_phones, dict) and ref in truth_phones and has_gt_times:
        mapped: list[float] = []
        for t0, _, _ in detections:
            n_loc = engine.to_local(ref, t0)
            if n_loc is None:
                mapped = []
                break
            mapped.append(_sim_true_time(truth_phones[ref], n_loc))
        if mapped:
            det_times = mapped
    elif has_gt_times:
        # Use direct timestamp matching only when CSV/meta timestamps share the engine's clock epoch.
        if any(abs(gt.t_s - t0) <= MATCH_WINDOW_S for gt in ground_truth for t0, _, _ in detections if gt.t_s is not None):
            det_times = [t0 for t0, _, _ in detections]

    if det_times is not None:
        matched: set[int] = set()
        errors: list[float] = []
        for t_det, (_, est_x, est_y) in zip(det_times, detections):
            idx = min(range(len(ground_truth)), key=lambda i: abs((ground_truth[i].t_s or 0.0) - t_det))
            gt = ground_truth[idx]
            if gt.t_s is None or abs(gt.t_s - t_det) > MATCH_WINDOW_S or idx in matched:
                continue
            matched.add(idx)
            errors.append(math.hypot(est_x - gt.x_m, est_y - gt.y_m))
        return errors

    # Chronological fallback when ground truth has no compatible timestamps.
    return [
        math.hypot(est_x - gt.x_m, est_y - gt.y_m)
        for (_, est_x, est_y), gt in zip(detections, ground_truth)
    ]


def _row_sort_key(row: TuneRow) -> tuple[bool, float, int]:
    return (
        row.mean_error_m is None,
        row.mean_error_m if row.mean_error_m is not None else float("inf"),
        row.missed,
    )


def sweep_recording(
    npz_path: str | Path,
    *,
    truth_csv: str | Path | None = None,
    grid: list[TuneParams] | None = None,
    max_combos: int = MAX_COMBOS,
) -> tuple[list[TuneRow], float]:
    """Sweep DSP parameter combinations on a recorded .npz session and return sorted results + wall-clock seconds."""
    if max_combos <= 0 or max_combos > MAX_COMBOS:
        raise ValueError(f"max_combos must be between 1 and {MAX_COMBOS}, got {max_combos}")
    combos = default_grid() if grid is None else list(grid)
    if len(combos) > max_combos:
        raise ValueError(f"Grid has {len(combos)} combinations, exceeding hard cap of {max_combos}")

    t_start = time.perf_counter()
    session = load_recording_npz(npz_path)
    ground_truth = load_ground_truth(session.meta, truth_csv=truth_csv)
    if not ground_truth:
        raise ValueError(f"No ground truth events found in '{npz_path}' or truth CSV")

    min_phones = min(3, len(session.phones))
    sync_cache: dict[tuple[float, float], SyncEngine] = {}
    onset_cache: dict[tuple[float, float, float, float], list[tuple[float, EventKind]]] = {}
    window_cache: dict[tuple[float, float, float, float, float], list[tuple[float, EventKind, dict[str, np.ndarray]]]] = {}

    rows: list[TuneRow] = []
    for params in combos:
        combo_t0 = time.perf_counter()

        sync_key = (params.chirp_direct_path_ratio, params.chirp_lookback_s)
        engine = sync_cache.get(sync_key)
        if engine is None:
            engine = SyncEngine(
                chirp_lookback_s=params.chirp_lookback_s,
                chirp_direct_path_ratio=params.chirp_direct_path_ratio,
            )
            engine.add_chirp_observation(session.slices, session.phones, session.room, "lfm_single")
            sync_cache[sync_key] = engine

        onset_key = (*sync_key, params.onset_snr_db, params.onset_refractory_s)
        grouped = onset_cache.get(onset_key)
        if grouped is None:
            grouped = _find_grouped_events(
                session,
                engine,
                onset_snr_db=params.onset_snr_db,
                onset_refractory_s=params.onset_refractory_s,
                min_phones=min_phones,
            )
            onset_cache[onset_key] = grouped

        win_key = (*onset_key, params.event_window_post_s)
        windows = window_cache.get(win_key)
        if windows is None:
            windows = _extract_windows(
                session,
                engine,
                grouped,
                event_window_post_s=params.event_window_post_s,
                min_phones=min_phones,
            )
            window_cache[win_key] = windows

        localizer = LocalizationPipeline(
            f_min=params.f_min,
            f_max=params.f_max,
            phat_beta=params.phat_beta,
            cycle_threshold_s=params.cycle_threshold_s,
            transient_post_s=params.event_window_post_s,
        )
        detections: list[tuple[float, float, float]] = []
        for t0, kind, aligned in windows:
            res = localizer.localize(
                aligned,
                {pid: session.phones[pid] for pid in aligned},
                session.room,
                kind,
                fs=TARGET_FS,
            )
            if res is not None:
                detections.append((t0, res.x_m, res.y_m))

        errors = _match_errors(detections, ground_truth, session.meta, engine)
        total = len(ground_truth)
        missed = total - len(errors)
        runtime_ms = (time.perf_counter() - combo_t0) * 1000.0
        rows.append(
            TuneRow(
                params=params,
                mean_error_m=float(np.mean(errors)) if errors else None,
                median_error_m=float(np.median(errors)) if errors else None,
                p90_error_m=float(np.percentile(errors, 90)) if errors else None,
                missed=missed,
                total=total,
                runtime_ms=runtime_ms,
            )
        )

    rows.sort(key=_row_sort_key)
    return rows, time.perf_counter() - t_start


def format_tuning_markdown(rows: list[TuneRow], *, recording_name: str, elapsed_s: float) -> str:
    """Format tuning results as a sorted Markdown table."""
    sorted_rows = sorted(rows, key=_row_sort_key)

    def fmt(val: float | None) -> str:
        return "-" if val is None else f"{val:.3f}"

    lines = [
        f"# DSP Tuning: {recording_name}",
        "",
        f"Wall-clock: {elapsed_s:.2f} s ({len(sorted_rows)} combinations)",
        "",
        "| rank | mean_m | median_m | p90_m | missed | runtime_ms | band_hz | beta | cycle_ms | post_ms | chirp_ratio | lookback_ms | snr_db | refrac_s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(sorted_rows, 1):
        p = row.params
        lines.append(
            f"| {rank} | {fmt(row.mean_error_m)} | {fmt(row.median_error_m)} | {fmt(row.p90_error_m)} | "
            f"{row.missed}/{row.total} | {row.runtime_ms:.1f} | "
            f"{p.f_min:.0f}-{p.f_max:.0f} | {p.phat_beta:.2f} | {p.cycle_threshold_s * 1000.0:.2f} | "
            f"{p.event_window_post_s * 1000.0:.0f} | {p.chirp_direct_path_ratio:.2f} | "
            f"{p.chirp_lookback_s * 1000.0:.0f} | {p.onset_snr_db:.1f} | {p.onset_refractory_s:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, grid: list[TuneParams] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded grid-search DSP parameter tuner for .npz recordings.")
    parser.add_argument("recording", help="Path to v2 .npz recording")
    parser.add_argument("--truth", default=None, help="Optional CSV with ground-truth positions")
    parser.add_argument("--out", default="analysis", help="Output directory for tuning-<stem>.md")
    parser.add_argument("--max-combos", type=int, default=MAX_COMBOS, help=f"Maximum combinations (<= {MAX_COMBOS})")
    args = parser.parse_args(argv)

    rec_path = Path(args.recording)
    rows, elapsed_s = sweep_recording(
        rec_path,
        truth_csv=args.truth,
        grid=grid,
        max_combos=args.max_combos,
    )
    md = format_tuning_markdown(rows, recording_name=rec_path.name, elapsed_s=elapsed_s)
    print(md)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tuning-{rec_path.stem}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
