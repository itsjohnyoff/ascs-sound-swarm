"""Single-event DSP diagnostic SVG plots: chirp envelopes, GCC-PHAT pairs, and triangle residuals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from itertools import combinations
import math
from pathlib import Path

import numpy as np

from backend.dsp.gcc_phat import compute_gcc_phat
from backend.dsp.pipeline import LocalizationPipeline
from backend.dsp.replay import RecordedSession, load_recording_npz
from backend.dsp.sync import SyncEngine, _compute_chirp_envelope, detect_all_chirps
from backend.server.events import TARGET_FS, find_events_offline
from shared.protocol import speed_of_sound


@dataclass(slots=True)
class EventDiagnostics:
    event_index: int
    t0_s: float
    kind: str
    est_xy: tuple[float, float] | None
    pair_tdoas_ms: dict[str, float]
    triangle_residuals_ms: dict[str, float]
    plot_paths: dict[str, Path]


def _svg_wrap(width: float, height: float, elements: list[str]) -> str:
    style = (
        "text{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#1f2937;}"
        ".title{font-size:13px;font-weight:600;}"
        ".suptitle{font-size:15px;font-weight:700;}"
        ".label{font-size:11px;fill:#374151;}"
        ".tick{font-size:10px;fill:#4b5563;}"
        ".legend{font-size:10px;fill:#1f2937;}"
    )
    body = "\n  ".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">\n'
        f"  <style>{style}</style>\n"
        f'  <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>\n'
        f"  {body}\n"
        f"</svg>\n"
    )


def _downsample_minmax(xs: np.ndarray, ys: np.ndarray, max_buckets: int = 350) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dense waveforms while preserving every local peak and trough."""
    n = len(xs)
    if n <= max_buckets * 2:
        return xs, ys
    edges = np.linspace(0, n, max_buckets + 1, dtype=np.intp)
    out_x: list[float] = []
    out_y: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = ys[a:b]
        i_min = a + int(np.argmin(seg))
        i_max = a + int(np.argmax(seg))
        if i_min <= i_max:
            out_x.extend((float(xs[i_min]), float(xs[i_max])))
            out_y.extend((float(ys[i_min]), float(ys[i_max])))
        else:
            out_x.extend((float(xs[i_max]), float(xs[i_min])))
            out_y.extend((float(ys[i_max]), float(ys[i_min])))
    return np.asarray(out_x, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


def _map_x(val: float, x_min: float, x_max: float, px0: float, px1: float) -> float:
    span = x_max - x_min if abs(x_max - x_min) > 1e-12 else 1.0
    return px0 + (val - x_min) / span * (px1 - px0)


def _map_y(val: float, y_min: float, y_max: float, py0: float, py1: float) -> float:
    span = y_max - y_min if abs(y_max - y_min) > 1e-12 else 1.0
    return py1 - (val - y_min) / span * (py1 - py0)


def _draw_axes(
    elems: list[str],
    px0: float,
    py0: float,
    px1: float,
    py1: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    nx_ticks: int = 5,
    ny_ticks: int = 4,
    x_ticks: bool = True,
) -> None:
    elems.append(f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{px1 - px0:.1f}" height="{py1 - py0:.1f}" fill="#fafafa" stroke="#d1d5db" stroke-width="1"/>')
    for k in range(ny_ticks + 1):
        frac = k / float(ny_ticks)
        val = y_min + frac * (y_max - y_min)
        y = _map_y(val, y_min, y_max, py0, py1)
        elems.append(f'<line x1="{px0:.1f}" y1="{y:.1f}" x2="{px1:.1f}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="0.8"/>')
        elems.append(f'<text class="tick" x="{px0 - 6:.1f}" y="{y + 3.5:.1f}" text-anchor="end">{val:.2g}</text>')

    if x_ticks:
        for k in range(nx_ticks + 1):
            frac = k / float(nx_ticks)
            val = x_min + frac * (x_max - x_min)
            x = _map_x(val, x_min, x_max, px0, px1)
            elems.append(f'<line x1="{x:.1f}" y1="{py0:.1f}" x2="{x:.1f}" y2="{py1:.1f}" stroke="#e5e7eb" stroke-width="0.8"/>')
            elems.append(f'<text class="tick" x="{x:.1f}" y="{py1 + 14:.1f}" text-anchor="middle">{val:.1f}</text>')

    elems.append(f'<text class="title" x="{(px0 + px1) * 0.5:.1f}" y="{py0 - 8:.1f}" text-anchor="middle">{escape(title)}</text>')
    if xlabel:
        elems.append(f'<text class="label" x="{(px0 + px1) * 0.5:.1f}" y="{py1 + 30:.1f}" text-anchor="middle">{escape(xlabel)}</text>')
    if ylabel:
        ymid = (py0 + py1) * 0.5
        elems.append(
            f'<text class="label" x="{px0 - 44:.1f}" y="{ymid:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 {px0 - 44:.1f} {ymid:.1f})">{escape(ylabel)}</text>'
        )


def _draw_legend(elems: list[str], x_right: float, y_top: float, items: list[tuple[str, str, str]]) -> None:
    """Draw compact top-right legend; items are (color, dasharray, label)."""
    if not items:
        return
    box_w = max(125.0, max(len(lbl) for _, _, lbl in items) * 5.8 + 30.0)
    box_h = 14.0 * len(items) + 8.0
    x0 = x_right - box_w
    elems.append(
        f'<rect x="{x0:.1f}" y="{y_top:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
        f'rx="3" fill="#ffffff" fill-opacity="0.88" stroke="#d1d5db" stroke-width="0.8"/>'
    )
    for idx, (color, dash, lbl) in enumerate(items):
        cy = y_top + 11.0 + idx * 14.0
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elems.append(
            f'<line x1="{x0 + 6:.1f}" y1="{cy:.1f}" x2="{x0 + 20:.1f}" y2="{cy:.1f}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>'
        )
        elems.append(f'<text class="legend" x="{x0 + 25:.1f}" y="{cy + 3.5:.1f}">{escape(lbl)}</text>')


def _plot_chirp_envelopes(
    session: RecordedSession,
    *,
    out_path: Path,
    chirp_direct_path_ratio: float,
    chirp_lookback_s: float,
) -> Path:
    pids = sorted(session.slices.keys())
    n_phones = max(1, len(pids))
    width = 860.0
    panel_h = 200.0
    header_h = 34.0
    height = header_h + panel_h * n_phones + 18.0
    elems: list[str] = [
        f'<text class="suptitle" x="{width * 0.5:.1f}" y="22" text-anchor="middle">'
        f"{escape(f'Chirp Matched-Filter Envelopes — {session.path.name}')}</text>"
    ]

    for row_idx, pid in enumerate(pids):
        top = header_h + row_idx * panel_h
        px0, px1 = 72.0, width - 24.0
        py0, py1 = top + 24.0, top + panel_h - 38.0
        slc = session.slices[pid]
        fs = slc.sample_rate
        valid_mask = getattr(slc, "valid_mask", None)
        prep = _compute_chirp_envelope(slc.samples_f32, fs=fs, valid_mask=valid_mask)
        arrivals = detect_all_chirps(
            slc.samples_f32,
            fs=fs,
            start_sample_index=slc.start_sample_index,
            valid_mask=valid_mask,
            lookback_s=chirp_lookback_s,
            direct_path_ratio=chirp_direct_path_ratio,
        )
        xlabel = "Time relative to local global peak (ms)" if row_idx == n_phones - 1 else ""
        if prep is None or not arrivals:
            _draw_axes(elems, px0, py0, px1, py1, -30.0, 30.0, 0.0, 1.0, title=f"{pid} (fs={fs} Hz)", xlabel=xlabel, ylabel="Envelope")
            elems.append(f'<text class="label" x="{(px0 + px1) * 0.5:.1f}" y="{(py0 + py1) * 0.5:.1f}" text-anchor="middle">{escape(pid)}: no chirp detected</text>')
            continue

        env = prep[0]
        direct_idx = arrivals[0].local_offset
        half_win = int(round(0.030 * fs))
        center_i = int(round(direct_idx))
        lo = max(0, center_i - half_win)
        hi = min(len(env), center_i + half_win + 1)
        seg = env[lo:hi]
        i_global = lo + int(np.argmax(seg))
        peak_val = float(env[i_global])

        t_ms = (np.arange(lo, hi, dtype=np.float64) - float(i_global)) / float(fs) * 1000.0
        direct_ms = (direct_idx - float(i_global)) / float(fs) * 1000.0
        lookback_ms = chirp_lookback_s * 1000.0
        x_min, x_max = float(t_ms[0]), float(t_ms[-1])
        y_min, y_max = 0.0, max(peak_val * 1.12, 1e-6)

        title = f"{pid} — first chirp (SNR={arrivals[0].snr_db:.1f} dB, NCC={arrivals[0].ncc:.2f})"
        _draw_axes(elems, px0, py0, px1, py1, x_min, x_max, y_min, y_max, title=title, xlabel=xlabel, ylabel="Envelope")

        lb_x0 = _map_x(max(x_min, -lookback_ms), x_min, x_max, px0, px1)
        lb_x1 = _map_x(min(x_max, 0.0), x_min, x_max, px0, px1)
        if lb_x1 > lb_x0:
            elems.append(f'<rect x="{lb_x0:.1f}" y="{py0:.1f}" width="{lb_x1 - lb_x0:.1f}" height="{py1 - py0:.1f}" fill="#f59e0b" fill-opacity="0.16"/>')

        ds_x, ds_y = _downsample_minmax(t_ms, seg)
        pts = " ".join(f"{_map_x(float(x), x_min, x_max, px0, px1):.1f},{_map_y(float(y), y_min, y_max, py0, py1):.1f}" for x, y in zip(ds_x, ds_y))
        elems.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="1.3" points="{pts}"/>')

        y_thr = _map_y(chirp_direct_path_ratio * peak_val, y_min, y_max, py0, py1)
        elems.append(f'<line x1="{px0:.1f}" y1="{y_thr:.1f}" x2="{px1:.1f}" y2="{y_thr:.1f}" stroke="#7c3aed" stroke-width="1.1" stroke-dasharray="5,3"/>')

        x_glob = _map_x(0.0, x_min, x_max, px0, px1)
        elems.append(f'<line x1="{x_glob:.1f}" y1="{py0:.1f}" x2="{x_glob:.1f}" y2="{py1:.1f}" stroke="#dc2626" stroke-width="1.3" stroke-dasharray="2,2"/>')

        x_dir = _map_x(float(np.clip(direct_ms, x_min, x_max)), x_min, x_max, px0, px1)
        elems.append(f'<line x1="{x_dir:.1f}" y1="{py0:.1f}" x2="{x_dir:.1f}" y2="{py1:.1f}" stroke="#16a34a" stroke-width="1.5"/>')

        _draw_legend(
            elems,
            px1 - 6.0,
            py0 + 6.0,
            [
                ("#2563eb", "", "Hilbert envelope"),
                ("#f59e0b", "", f"lookback ({lookback_ms:.0f} ms)"),
                ("#7c3aed", "5,3", f"thresh ({chirp_direct_path_ratio:.2f}x)"),
                ("#dc2626", "2,2", "global peak (0 ms)"),
                ("#16a34a", "", f"direct ({direct_ms:+.2f} ms)"),
            ],
        )

    out_path.write_text(_svg_wrap(width, height, elems), encoding="utf-8")
    return out_path


def _plot_gcc_pairs(
    session: RecordedSession,
    aligned: dict[str, np.ndarray],
    est_xy: tuple[float, float] | None,
    *,
    event_index: int,
    out_path: Path,
    f_min: float,
    f_max: float,
    phat_beta: float,
) -> tuple[dict[str, float], Path]:
    pids = sorted(pid for pid in aligned if pid in session.phones)
    pairs = list(combinations(pids, 2))
    c = speed_of_sound(session.room.temperature_c)

    pair_tdoas_ms: dict[str, float] = {}
    n_pairs = max(1, len(pairs))
    ncols = min(3, n_pairs)
    nrows = math.ceil(n_pairs / ncols)
    cell_w, cell_h = 340.0, 230.0
    header_h = 50.0
    width = ncols * cell_w + 16.0
    height = header_h + nrows * cell_h + 10.0

    est_str = f"({est_xy[0]:.2f}, {est_xy[1]:.2f}) m" if est_xy is not None else "N/A"
    elems: list[str] = [
        f'<text class="suptitle" x="{width * 0.5:.1f}" y="22" text-anchor="middle">'
        f"{escape(f'Event #{event_index} GCC-PHAT Pairs')}</text>",
        f'<text class="label" x="{width * 0.5:.1f}" y="40" text-anchor="middle">'
        f"{escape(f'Raw aligned window — est={est_str} (c={c:.1f} m/s)')}</text>",
    ]

    if not pairs:
        _draw_axes(elems, 64.0, header_h + 24.0, width - 24.0, height - 42.0, -1.0, 1.0, -1.0, 1.0, title="GCC-PHAT", xlabel="Lag (ms)", ylabel="Norm. GCC-PHAT")
        elems.append(f'<text class="label" x="{width * 0.5:.1f}" y="{height * 0.5:.1f}" text-anchor="middle">Fewer than 2 aligned phones</text>')

    for idx, (pi, pj) in enumerate(pairs):
        col, row = idx % ncols, idx // ncols
        ox, oy = col * cell_w, header_h + row * cell_h
        px0, px1 = ox + 60.0, ox + cell_w - 16.0
        py0, py1 = oy + 26.0, oy + cell_h - 40.0

        ph_i, ph_j = session.phones[pi], session.phones[pj]
        d_ij = math.hypot(ph_i.x_m - ph_j.x_m, ph_i.y_m - ph_j.y_m)
        max_tau_s = d_ij / c + 0.0005

        lags_s, corr, best_tau_s, psr_w = compute_gcc_phat(
            aligned[pi],
            aligned[pj],
            fs=TARGET_FS,
            max_tau_s=max_tau_s,
            f_min=f_min,
            f_max=f_max,
            phat_beta=phat_beta,
        )
        peak = float(np.max(np.abs(corr))) if len(corr) else 0.0
        norm_corr = corr / peak if peak > 1e-12 else corr
        lags_ms = lags_s * 1000.0
        meas_ms = best_tau_s * 1000.0
        pair_tdoas_ms[f"{pi}-{pj}"] = meas_ms

        x_min, x_max = float(lags_ms[0]), float(lags_ms[-1])
        y_min = min(-0.25, float(np.min(norm_corr)) * 1.1) if len(norm_corr) else -0.25
        y_max = max(1.1, float(np.max(norm_corr)) * 1.1) if len(norm_corr) else 1.1

        _draw_axes(
            elems, px0, py0, px1, py1, x_min, x_max, y_min, y_max,
            title=f"{pi}-{pj} (PSR w={psr_w:.2f})",
            xlabel="Lag (ms)",
            ylabel="Norm. GCC-PHAT" if col == 0 else "",
        )

        ds_x, ds_y = _downsample_minmax(lags_ms, norm_corr, max_buckets=220)
        pts = " ".join(f"{_map_x(float(x), x_min, x_max, px0, px1):.1f},{_map_y(float(y), y_min, y_max, py0, py1):.1f}" for x, y in zip(ds_x, ds_y))
        elems.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="1.2" points="{pts}"/>')

        x_meas = _map_x(float(np.clip(meas_ms, x_min, x_max)), x_min, x_max, px0, px1)
        elems.append(f'<line x1="{x_meas:.1f}" y1="{py0:.1f}" x2="{x_meas:.1f}" y2="{py1:.1f}" stroke="#dc2626" stroke-width="1.4"/>')
        legend_items = [("#dc2626", "", f"meas {meas_ms:+.2f} ms")]

        if est_xy is not None:
            d_i = math.hypot(est_xy[0] - ph_i.x_m, est_xy[1] - ph_i.y_m)
            d_j = math.hypot(est_xy[0] - ph_j.x_m, est_xy[1] - ph_j.y_m)
            pred_ms = (d_i - d_j) / c * 1000.0
            x_pred = _map_x(float(np.clip(pred_ms, x_min, x_max)), x_min, x_max, px0, px1)
            elems.append(f'<line x1="{x_pred:.1f}" y1="{py0:.1f}" x2="{x_pred:.1f}" y2="{py1:.1f}" stroke="#16a34a" stroke-width="1.4" stroke-dasharray="5,3"/>')
            legend_items.append(("#16a34a", "5,3", f"pred {pred_ms:+.2f} ms"))

        _draw_legend(elems, px1 - 4.0, py0 + 4.0, legend_items)

    out_path.write_text(_svg_wrap(width, height, elems), encoding="utf-8")
    return pair_tdoas_ms, out_path


def _plot_triangle_residuals(
    pids: list[str],
    pair_tdoas_ms: dict[str, float],
    *,
    event_index: int,
    out_path: Path,
    cycle_threshold_s: float,
) -> tuple[dict[str, float], Path]:
    triples = list(combinations(pids, 3))
    triangle_residuals_ms: dict[str, float] = {}
    for pi, pj, pk in triples:
        tau_ij = pair_tdoas_ms[f"{pi}-{pj}"]
        tau_jk = pair_tdoas_ms[f"{pj}-{pk}"]
        tau_ik = pair_tdoas_ms[f"{pi}-{pk}"]
        triangle_residuals_ms[f"{pi}-{pj}-{pk}"] = abs(tau_ij + tau_jk - tau_ik)

    width, height = 740.0, 360.0
    px0, py0, px1, py1 = 76.0, 56.0, width - 28.0, height - 56.0
    elems: list[str] = [
        f'<text class="suptitle" x="{width * 0.5:.1f}" y="24" text-anchor="middle">'
        f"{escape(f'Event #{event_index} Triangle Residuals — Raw aligned window')}</text>"
    ]

    if not triangle_residuals_ms:
        elems.append(f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{px1 - px0:.1f}" height="{py1 - py0:.1f}" fill="#fafafa" stroke="#d1d5db" stroke-width="1"/>')
        msg = f"Requires >= 3 phones for triangle residuals (have {len(pids)})"
        elems.append(f'<text class="title" x="{(px0 + px1) * 0.5:.1f}" y="{(py0 + py1) * 0.5:.1f}" text-anchor="middle">{escape(msg)}</text>')
    else:
        labels = list(triangle_residuals_ms.keys())
        values = [triangle_residuals_ms[k] for k in labels]
        thresh_ms = cycle_threshold_s * 1000.0
        y_max = max(max(values) * 1.25, thresh_ms * 1.15, 0.75)

        _draw_axes(
            elems, px0, py0, px1, py1, 0.0, float(len(labels)), 0.0, y_max,
            title="", xlabel="Phone triple", ylabel="|tau_ij + tau_jk - tau_ik| (ms)", x_ticks=False,
        )

        slot_w = (px1 - px0) / float(len(labels))
        bar_w = min(64.0, slot_w * 0.55)
        for idx, (lbl, val) in enumerate(zip(labels, values)):
            cx = px0 + (idx + 0.5) * slot_w
            bx = cx - bar_w * 0.5
            by = _map_y(val, 0.0, y_max, py0, py1)
            color = "#dc2626" if val > thresh_ms else "#2563eb"
            elems.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(1.0, py1 - by):.1f}" fill="{color}" fill-opacity="0.85"/>')
            elems.append(f'<text class="tick" x="{cx:.1f}" y="{by - 4:.1f}" text-anchor="middle">{val:.3f}</text>')
            elems.append(f'<text class="tick" x="{cx:.1f}" y="{py1 + 15:.1f}" text-anchor="middle">{escape(lbl)}</text>')

        y_ref = _map_y(0.5, 0.0, y_max, py0, py1)
        elems.append(f'<line x1="{px0:.1f}" y1="{y_ref:.1f}" x2="{px1:.1f}" y2="{y_ref:.1f}" stroke="#16a34a" stroke-width="1.3" stroke-dasharray="6,3"/>')

        y_thr = _map_y(thresh_ms, 0.0, y_max, py0, py1)
        elems.append(f'<line x1="{px0:.1f}" y1="{y_thr:.1f}" x2="{px1:.1f}" y2="{y_thr:.1f}" stroke="#f59e0b" stroke-width="1.3" stroke-dasharray="6,3"/>')

        _draw_legend(
            elems,
            px1 - 8.0,
            py0 + 8.0,
            [
                ("#16a34a", "6,3", "0.5 ms reference"),
                ("#f59e0b", "6,3", f"Localizer threshold for reference ({thresh_ms:.2f} ms)"),
            ],
        )

    out_path.write_text(_svg_wrap(width, height, elems), encoding="utf-8")
    return triangle_residuals_ms, out_path


def generate_event_diagnostics(
    npz_path: str | Path,
    *,
    event_index: int = 0,
    out_dir: str | Path = "analysis/plots",
    chirp_direct_path_ratio: float = 0.55,
    chirp_lookback_s: float = 0.025,
    f_min: float = 300.0,
    f_max: float = 6000.0,
    phat_beta: float = 0.8,
    cycle_threshold_s: float = 0.0015,
) -> EventDiagnostics:
    """Run offline detection/localization and render 3 diagnostic SVGs for a single event."""
    session = load_recording_npz(npz_path)
    engine = SyncEngine(
        chirp_lookback_s=chirp_lookback_s,
        chirp_direct_path_ratio=chirp_direct_path_ratio,
    )
    engine.add_chirp_observation(session.slices, session.phones, session.room, "lfm_single")

    localizer = LocalizationPipeline(
        f_min=f_min,
        f_max=f_max,
        phat_beta=phat_beta,
        cycle_threshold_s=cycle_threshold_s,
    )
    min_phones = max(2, min(3, len(session.phones)))
    detections = find_events_offline(
        session.slices,
        session.phones,
        session.room,
        engine=engine,
        localizer=localizer,
        min_phones=min_phones,
    )
    if event_index < 0 or event_index >= len(detections):
        raise IndexError(f"event_index {event_index} out of range (found {len(detections)} events)")

    det = detections[event_index]
    est_xy = (det.result.x_m, det.result.y_m) if det.result is not None else None
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(npz_path).stem

    chirp_svg = _plot_chirp_envelopes(
        session,
        out_path=out_path / f"{stem}-event{event_index}-chirp-envelope.svg",
        chirp_direct_path_ratio=chirp_direct_path_ratio,
        chirp_lookback_s=chirp_lookback_s,
    )
    pair_tdoas_ms, gcc_svg = _plot_gcc_pairs(
        session,
        det.aligned,
        est_xy,
        event_index=event_index,
        out_path=out_path / f"{stem}-event{event_index}-gcc-pairs.svg",
        f_min=f_min,
        f_max=f_max,
        phat_beta=phat_beta,
    )
    aligned_pids = sorted(pid for pid in det.aligned if pid in session.phones)
    triangle_residuals_ms, tri_svg = _plot_triangle_residuals(
        aligned_pids,
        pair_tdoas_ms,
        event_index=event_index,
        out_path=out_path / f"{stem}-event{event_index}-triangle-residuals.svg",
        cycle_threshold_s=cycle_threshold_s,
    )

    return EventDiagnostics(
        event_index=event_index,
        t0_s=det.t0,
        kind=det.kind,
        est_xy=est_xy,
        pair_tdoas_ms=pair_tdoas_ms,
        triangle_residuals_ms=triangle_residuals_ms,
        plot_paths={
            "chirp_envelope": chirp_svg,
            "gcc_pairs": gcc_svg,
            "triangle_residuals": tri_svg,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate single-event DSP diagnostic SVG plots.")
    parser.add_argument("recording", help="Path to v2 .npz recording")
    parser.add_argument("--event", type=int, default=0, help="0-based detected event index (default: 0)")
    parser.add_argument("--out", default="analysis/plots", help="Output directory for SVG plots")
    args = parser.parse_args(argv)

    diag = generate_event_diagnostics(args.recording, event_index=args.event, out_dir=args.out)
    max_tri = max(diag.triangle_residuals_ms.values()) if diag.triangle_residuals_ms else None
    tri_str = f"{max_tri:.3f} ms" if max_tri is not None else "N/A (<3 phones)"
    print(f"Event #{diag.event_index} ({diag.kind}, t0={diag.t0_s:.3f}s) est_xy={diag.est_xy} raw-window max_tri={tri_str}")
    for key, path in diag.plot_paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
