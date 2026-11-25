"""Gate 0 spike report: python -m analysis.spike data/recordings/spike-01.npz

Per phone: actual sample rate, gaps in sample_index, chirps found, clock-fit residual,
drift. Per clap: measured TDOA per pair vs TDOA predicted from the taped clap point.
Clap points come from an optional --clap-points JSON file, meta["clap_points"]
([{x_m, y_m}], in clap order), or meta["truth"]["events"] in simulation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from backend.dsp.gcc_phat import compute_gcc_phat
from backend.dsp.pipeline import SyncEngine
from backend.dsp.replay import load_recording_npz
from backend.server.events import TARGET_FS, find_events_offline
from shared.protocol import PhoneAudioSlice, speed_of_sound


@dataclass
class PhoneReport:
    fs: int
    gap_count: int
    chirps: int
    residual_ms: float | None
    drift_ppm: float | None
    clipped_fraction: float
    fit_chirps: int
    agc_gated: bool = False
    agc_reason: str = ""
    dropped_frames: int | None = None
    segments: int = 1
    disconnects: int = 0


@dataclass
class ClapReport:
    t_global_s: float
    point: tuple[float, float] | None
    pair_errors_ms: dict[str, float] = field(default_factory=dict)

    @property
    def max_tdoa_error_ms(self) -> float:
        return max(self.pair_errors_ms.values(), default=math.nan)


@dataclass
class SpikeReport:
    path: str
    ref_phone_id: str | None
    phones: dict[str, PhoneReport]
    claps: list[ClapReport]
    taped_claps: int = 0

    @property
    def gate_failures(self) -> list[str]:
        failures = []
        if not 2 <= len(self.phones) <= 4:
            failures.append(f"Capture needs 2–4 phones with audio, got {len(self.phones)}")
        for pid, phone in sorted(self.phones.items()):
            if phone.chirps < 3 or phone.fit_chirps < 3:
                failures.append(f"{pid}: only {phone.fit_chirps} chirps in the clock fit ({phone.chirps} detected), need 3")
            if phone.residual_ms is None or not phone.residual_ms < 1.0:
                failures.append(f"{pid}: residual {phone.residual_ms} ms, need < 1 ms")
            if phone.gap_count:
                failures.append(f"{pid}: {phone.gap_count} sample-index gap(s)")
            if phone.dropped_frames:
                failures.append(f"{pid}: {phone.dropped_frames} dropped frame(s)")
            if phone.segments > 1 or phone.disconnects:
                failures.append(f"{pid}: discontinuous capture ({phone.segments} segments, {phone.disconnects} disconnects)")
            if phone.clipped_fraction >= 0.01:
                failures.append(f"{pid}: {100 * phone.clipped_fraction:.2f}% clipped, need < 1%")
            if phone.agc_gated:
                failures.append(f"{pid}: AGC/gate flagged ({phone.agc_reason})")
        return failures


def _gap_count(valid: np.ndarray | None) -> int:
    if valid is None or valid.all():
        return 0
    invalid = (~valid).astype(np.int8)
    return int(np.sum(np.diff(np.concatenate([[0], invalid])) == 1))


def _detect_agc_or_gating(
    samples: np.ndarray,
    fs: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[bool, str]:
    frame_len = max(1, int(round(0.02 * fs)))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return False, ""
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len).astype(np.float64, copy=False)
    frame_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    if valid_mask is not None:
        frame_valid = valid_mask[: n_frames * frame_len].reshape(n_frames, frame_len).all(axis=1)
    else:
        frame_valid = np.ones(n_frames, dtype=bool)
    valid_rms = frame_rms[frame_valid]
    if len(valid_rms) == 0:
        return False, ""

    peak = float(np.max(np.abs(trimmed[frame_valid])))
    if peak >= 0.02 and float(np.mean(valid_rms < 1e-5)) > 0.10:
        return True, "noise-gated"

    thresh = max(8.0 * float(np.median(valid_rms)), 0.03)
    cand = np.flatnonzero(frame_valid & (frame_rms > thresh))
    min_sep = max(1, int(round(1.0 / 0.02)))
    peaks: list[int] = []
    for idx in cand[np.argsort(frame_rms[cand])[::-1]]:
        p = int(idx)
        if all(abs(p - kept) >= min_sep for kept in peaks):
            peaks.append(p)

    lo_off = int(round(0.15 / 0.02))
    hi_off = int(round(0.45 / 0.02))
    dipped = 0
    for p in peaks:
        if p - hi_off < 0 or p + hi_off > n_frames:
            continue
        pre = slice(p - hi_off, p - lo_off)
        post = slice(p + lo_off, p + hi_off)
        if not (frame_valid[pre].all() and frame_valid[post].all()):
            continue
        pre_rms = float(np.median(frame_rms[pre]))
        post_rms = float(np.median(frame_rms[post]))
        if pre_rms > 0.0 and post_rms < 0.45 * pre_rms:
            dipped += 1
    if dipped >= 2:
        return True, "agc-ducking"
    return False, ""


def _clap_points(meta: dict) -> list[tuple[float, float]]:
    if meta.get("clap_points"):
        return [(float(p["x_m"]), float(p["y_m"])) for p in meta["clap_points"]]
    events = meta.get("truth", {}).get("events", [])
    return [(e["x_m"], e["y_m"]) for e in events if e["kind"] == "clap"]


def analyze(path: str | Path, *, clap_points: list[tuple[float, float]] | None = None) -> SpikeReport:
    session = load_recording_npz(path)
    slices = {pid: PhoneAudioSlice(pid, s.samples_f32, s.start_sample_index, s.sample_rate, s.valid_mask)
              for pid, s in session.slices.items()}
    engine = SyncEngine()
    engine.add_chirp_observation(slices, session.phones, session.room, "lfm_single")
    chirps = {pid: 0 for pid in slices}
    for em in engine._v2_fitter.emissions:
        for pid in em.obs_by_phone:
            chirps[pid] += 1

    phones = {}
    for pid, s in slices.items():
        model = engine.clock(pid)
        agc_gated, agc_reason = _detect_agc_or_gating(s.samples_f32, s.sample_rate, s.valid_mask)
        stats = session.meta.get("frame_stats", {}).get(pid, {})
        segments = session.meta.get("segments", {}).get(pid, ())
        valid_samples = s.samples_f32 if s.valid_mask is None else s.samples_f32[s.valid_mask]
        phones[pid] = PhoneReport(
            fs=s.sample_rate, gap_count=max(_gap_count(s.valid_mask), stats.get("sample_gaps", 0)),
            chirps=chirps.get(pid, 0), fit_chirps=0 if model is None else model.n_points,
            residual_ms=None if model is None else model.residual_ms,
            drift_ppm=None if model is None else round(model.drift_ppm, 2),
            clipped_fraction=float(np.mean(np.abs(valid_samples) >= 0.99)) if len(valid_samples) else 0.0,
            agc_gated=agc_gated, agc_reason=agc_reason,
            dropped_frames=stats.get("dropped_frames"), segments=stats.get("segments", len(segments) or 1),
            disconnects=stats.get("disconnects", 0),
        )

    c = speed_of_sound(session.room.temperature_c)
    points = _clap_points(session.meta) if clap_points is None else clap_points
    claps = []
    detections = [d for d in find_events_offline(slices, session.phones, session.room, engine=engine, min_phones=2)
                  if d.kind == "transient"]
    # Assigning by order after an extra or missed transient could pair a clap with the wrong point.
    points_match = len(points) == len(detections)
    for i, det in enumerate(detections):
        point = points[i] if points_match else None
        rep = ClapReport(det.t0, point)
        if point is not None:
            for a, b in combinations(sorted(det.aligned), 2):
                pa, pb = session.phones[a], session.phones[b]
                da = math.hypot(pa.x_m - point[0], pa.y_m - point[1])
                db = math.hypot(pb.x_m - point[0], pb.y_m - point[1])
                max_tau = math.hypot(pa.x_m - pb.x_m, pa.y_m - pb.y_m) / c + 0.0005
                _, _, tau, _ = compute_gcc_phat(det.aligned[a], det.aligned[b], fs=TARGET_FS, max_tau_s=max_tau)
                rep.pair_errors_ms[f"{a}-{b}"] = round(abs(tau - (da - db) / c) * 1e3, 4)
        claps.append(rep)
    return SpikeReport(str(path), engine.ref_phone_id, phones, claps, taped_claps=len(points))


def format_report(r: SpikeReport) -> str:
    lines = [f"# Spike report: {r.path}", "", f"Reference phone: {r.ref_phone_id}", "",
             "| phone | fs | gaps | chirps detected/fit | residual ms | drift ppm | clipped % | AGC/gate |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for pid, p in sorted(r.phones.items()):
        agc_col = f"FLAG ({p.agc_reason})" if p.agc_gated else "OK"
        residual = "—" if p.residual_ms is None else f"{p.residual_ms:.4f}"
        lines.append(f"| {pid} | {p.fs} | {p.gap_count} | {p.chirps}/{p.fit_chirps} | {residual} | {p.drift_ppm} | "
                     f"{100 * p.clipped_fraction:.2f} | {agc_col} |")
    lines += ["", "| clap | t global s | point | max pair TDOA error ms |", "| --- | --- | --- | --- |"]
    for i, c in enumerate(r.claps, 1):
        lines.append(f"| {i} | {c.t_global_s:.3f} | {c.point} | {c.max_tdoa_error_ms:.3f} |")
    if r.taped_claps and r.taped_claps != len(r.claps):
        lines += ["", f"TDOA comparison needs matching counts. Found {len(r.claps)} transient events for {r.taped_claps} taped claps."]
    failures = r.gate_failures
    lines += ["", f"Gate 0: {'FAIL' if failures else 'PASS'}"]
    if failures:
        lines += ["", *(f"- {failure}" for failure in failures)]
    return "\n".join(lines)


def _plot(r: SpikeReport, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    session = load_recording_npz(r.path)
    fig, axes = plt.subplots(len(session.slices), 1, figsize=(10, 2 * len(session.slices)), sharex=False)
    for ax, (pid, s) in zip(np.atleast_1d(axes), sorted(session.slices.items())):
        ax.plot(np.arange(len(s.samples_f32)) / s.sample_rate, s.samples_f32, lw=0.3)
        ax.set_title(pid)
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(r.path).stem}-waveforms.png", dpi=80)


def _read_clap_points(path: Path) -> list[tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("expected an array of {x_m, y_m} points in clap order")
    points = []
    for i, point in enumerate(raw, 1):
        xy = [point.get(axis) for axis in ("x_m", "y_m")] if isinstance(point, dict) else []
        if len(xy) != 2 or any(type(n) not in (int, float) or not math.isfinite(n) for n in xy):
            raise ValueError(f"point {i} needs finite numeric x_m and y_m coordinates")
        points.append((float(xy[0]), float(xy[1])))
    return points


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--clap-points", type=Path, help="JSON array of {x_m, y_m} taped points in clap order")
    ap.add_argument("--plots", default="analysis/plots")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        points = _read_clap_points(args.clap_points) if args.clap_points is not None else None
    except (OSError, ValueError, OverflowError) as exc:
        ap.error(f"--clap-points: {exc}")
    report = analyze(args.path, clap_points=points)
    if args.json:
        print(json.dumps({**asdict(report), "gate_failures": report.gate_failures}, indent=2))
    else:
        print(format_report(report))
    _plot(report, Path(args.plots))
    return int(bool(report.gate_failures))


if __name__ == "__main__":
    raise SystemExit(main())
