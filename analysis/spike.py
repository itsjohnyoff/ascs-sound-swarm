"""Gate 0 spike report: python -m analysis.spike data/recordings/spike-01.npz

Per phone: actual sample rate, gaps in sample_index, chirps found, clock-fit residual,
drift. Per clap: measured TDOA per pair vs TDOA predicted from the taped clap point.
Clap points come from meta["clap_points"] ([{x_m, y_m}], in clap order) or, for
simulated files, from meta["truth"]["events"].
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
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
    agc_gated: bool = False
    agc_reason: str = ""


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


def analyze(path: str | Path) -> SpikeReport:
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
        phones[pid] = PhoneReport(
            fs=s.sample_rate, gap_count=_gap_count(s.valid_mask), chirps=chirps.get(pid, 0),
            residual_ms=None if model is None else round(model.residual_ms, 4),
            drift_ppm=None if model is None else round(model.drift_ppm, 2),
            clipped_fraction=float(np.mean(np.abs(s.samples_f32) >= 0.99)),
            agc_gated=agc_gated, agc_reason=agc_reason,
        )

    c = speed_of_sound(session.room.temperature_c)
    points = _clap_points(session.meta)
    claps = []
    detections = [d for d in find_events_offline(slices, session.phones, session.room, engine=engine, min_phones=2)
                  if d.kind == "transient"]
    for i, det in enumerate(detections):
        point = points[i] if i < len(points) else None
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
    return SpikeReport(str(path), engine.ref_phone_id, phones, claps)


def format_report(r: SpikeReport) -> str:
    lines = [f"# Spike report: {r.path}", "", f"Reference phone: {r.ref_phone_id}", "",
             "| phone | fs | gaps | chirps | residual ms | drift ppm | clipped % | AGC/gate |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for pid, p in sorted(r.phones.items()):
        agc_col = f"FLAG ({p.agc_reason})" if p.agc_gated else "OK"
        lines.append(f"| {pid} | {p.fs} | {p.gap_count} | {p.chirps} | {p.residual_ms} | {p.drift_ppm} | "
                     f"{100 * p.clipped_fraction:.2f} | {agc_col} |")
    lines += ["", "| clap | t global s | point | max pair TDOA error ms |", "| --- | --- | --- | --- |"]
    for i, c in enumerate(r.claps, 1):
        lines.append(f"| {i} | {c.t_global_s:.3f} | {c.point} | {c.max_tdoa_error_ms:.3f} |")
    residuals = [p.residual_ms for p in r.phones.values() if p.residual_ms is not None]
    ok = bool(residuals) and len(residuals) == len(r.phones) and max(residuals) < 1.0
    lines += ["", f"Gate 0 (residual < 1 ms on every phone): {'PASS' if ok else 'FAIL'}"]
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--plots", default="analysis/plots")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = analyze(args.path)
    if args.json:
        print(json.dumps(report, default=lambda o: o.__dict__, indent=2))
    else:
        print(format_report(report))
    _plot(report, Path(args.plots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
