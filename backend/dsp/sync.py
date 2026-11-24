"""Sub-millisecond chirp synchronization, first-arrival echo defense, and PPM drift alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Literal

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import map_coordinates
from scipy.signal import butter, fftconvolve, find_peaks, hilbert, sosfiltfilt

from shared.protocol import (
    CHIRP_DURATION_S,
    DUAL_CHIRP_GAP_S,
    CalibrationMode,
    CalibrationResult,
    PhoneAudioSlice,
    PhoneConfig,
    PhoneSyncReport,
    RoomConfig,
    speed_of_sound,
)
from backend.dsp.chirp import generate_lfm_chirp
from backend.dsp.gcc_phat import compute_gcc_phat


def _parabolic_refine(arr: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    y_m1 = float(arr[idx - 1])
    y_0 = float(arr[idx])
    y_p1 = float(arr[idx + 1])
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-18:
        return float(idx)
    delta = float(np.clip(0.5 * (y_m1 - y_p1) / denom, -0.5, 0.5))
    return float(idx) + delta


def _local_template_ncc(
    env: np.ndarray,
    x_filt: np.ndarray,
    idx: int,
    tpl_len: int,
) -> float:
    """Normalized correlation magnitude against unit-norm chirp template at idx."""
    seg = x_filt[idx : idx + tpl_len]
    local_norm = float(np.linalg.norm(seg))
    if local_norm <= 1e-12:
        return 0.0
    return float(env[idx]) / local_norm


def _pick_first_arrival(
    env: np.ndarray,
    x_filt: np.ndarray,
    tpl_len: int,
    fs: int,
    start: int,
    end: int,
    sigma: float,
    min_snr_db: float,
    min_ncc: float,
) -> tuple[float | None, float]:
    """Locate earliest direct-path peak in env[start:end] matching the chirp template."""
    start = max(0, start)
    end = min(len(env), end)
    if end - start < 3:
        return None, -100.0

    region = env[start:end]
    rel_max = int(np.argmax(region))
    i_max = start + rel_max
    peak_val = float(env[i_max])
    snr_db = 20.0 * math.log10(max(peak_val, 1e-12) / max(sigma, 1e-12))
    if snr_db < min_snr_db:
        return None, snr_db

    # Search backwards 25 ms before strongest peak to recover direct path even if a wall echo is louder
    lookback = int(round(0.025 * fs))
    win_start = max(start, i_max - lookback)
    sub = env[win_start : i_max + 1]
    peaks, _ = find_peaks(sub, height=0.55 * peak_val, distance=max(1, int(0.001 * fs)))
    for pk in peaks:
        cand = win_start + int(pk)
        if _local_template_ncc(env, x_filt, cand, tpl_len) >= min_ncc:
            return _parabolic_refine(env, cand), snr_db

    if _local_template_ncc(env, x_filt, i_max, tpl_len) >= min_ncc:
        return _parabolic_refine(env, i_max), snr_db
    return None, snr_db


def detect_chirp_arrivals(
    samples_f32: np.ndarray,
    fs: int,
    mode: CalibrationMode = "lfm_dual",
    gap_s: float = DUAL_CHIRP_GAP_S,
    min_snr_db: float = 12.0,
    min_ncc: float = 0.25,
) -> tuple[list[float], float]:
    """Detect fractional chirp start sample indices inside samples_f32."""
    tpl = generate_lfm_chirp(fs=fs).astype(np.float64)
    if len(samples_f32) < len(tpl) + 16:
        return [], -100.0

    tpl_norm = float(np.linalg.norm(tpl))
    if tpl_norm < 1e-12:
        return [], -100.0
    tpl /= tpl_norm

    hi = min(8500.0, 0.48 * fs)
    sos = butter(4, [1200.0, hi], btype="bandpass", fs=fs, output="sos")
    x_filt = sosfiltfilt(sos, samples_f32.astype(np.float64))

    corr = fftconvolve(x_filt, tpl[::-1], mode="valid")
    n_fast = next_fast_len(len(corr))
    env = np.abs(hilbert(corr, N=n_fast)[: len(corr)])

    sigma = float(np.median(env)) / 1.17741 + 1e-12
    tpl_len = len(tpl)
    p_a, snr_a = _pick_first_arrival(env, x_filt, tpl_len, fs, 0, len(env), sigma, min_snr_db, min_ncc)
    if p_a is None:
        return [], snr_a

    if mode == "lfm_single":
        return [p_a], snr_a

    tol = int(round(0.08 * fs))
    gap_n = int(round(gap_s * fs))
    partner_min_snr = max(9.0, min_snr_db - 3.0)

    fwd_s = int(round(p_a)) + gap_n - tol
    fwd_e = int(round(p_a)) + gap_n + tol
    p_fwd, snr_fwd = _pick_first_arrival(env, x_filt, tpl_len, fs, fwd_s, fwd_e, sigma, partner_min_snr, min_ncc)

    bwd_s = int(round(p_a)) - gap_n - tol
    bwd_e = int(round(p_a)) - gap_n + tol
    p_bwd, snr_bwd = _pick_first_arrival(env, x_filt, tpl_len, fs, bwd_s, bwd_e, sigma, partner_min_snr, min_ncc)

    candidates: list[tuple[float, float]] = []
    if p_fwd is not None:
        candidates.append((p_fwd, snr_fwd))
    if p_bwd is not None:
        candidates.append((p_bwd, snr_bwd))

    if not candidates:
        return [p_a], snr_a

    p_b, snr_b = max(candidates, key=lambda item: item[1])
    arrivals = sorted([p_a, p_b])
    return arrivals, min(snr_a, snr_b)


@dataclass
class _ClockModel:
    a_s: float  # global time intercept at sample 0
    b_s: float  # seconds per local sample (1 / effective_fs)
    fs_nom: int
    drift_ppm: float
    chirp_snr_db: float
    residual_ms: float = 0.0
    last_sync_ts: float = 0.0
    synced: bool = True
    cal_points: list[tuple[float, float]] = field(default_factory=list)  # (n_abs, T_emit + tau_prop)


class SyncEngine:
    """Acoustic STO + SRO (clock drift) calibration and sub-sample timeline aligner."""

    def __init__(self) -> None:
        self._models: dict[str, _ClockModel] = {}

    def is_synced(self, phone_id: str) -> bool:
        model = self._models.get(phone_id)
        return bool(model is not None and model.synced)

    def mark_clock_jump(self, phone_id: str) -> None:
        model = self._models.get(phone_id)
        if model is not None:
            model.synced = False
            model.cal_points.clear()

    def sample_to_global_time(self, phone_id: str, sample_index: float) -> float:
        model = self._models.get(phone_id)
        if model is None or not model.synced:
            raise KeyError(f"Phone '{phone_id}' is not synced")
        return model.a_s + model.b_s * float(sample_index)

    def get_common_time_range(
        self,
        phone_buffers: dict[str, PhoneAudioSlice],
    ) -> tuple[float, float] | None:
        """Return overlapping [t_start, t_end] global seconds across all synced phones in buffers."""
        starts: list[float] = []
        ends: list[float] = []
        for pid, slc in phone_buffers.items():
            if not self.is_synced(pid) or len(slc.samples_f32) == 0:
                continue
            t0 = self.sample_to_global_time(pid, slc.start_sample_index)
            t1 = self.sample_to_global_time(pid, slc.start_sample_index + len(slc.samples_f32) - 1)
            starts.append(t0)
            ends.append(t1)
        if len(starts) < 2:
            return None
        lo = max(starts)
        hi = min(ends)
        return (lo, hi) if hi > lo else None

    def extract_aligned_window(
        self,
        phone_buffers: dict[str, PhoneAudioSlice],
        t_start_s: float,
        t_end_s: float,
        target_fs: int = 48000,
    ) -> dict[str, np.ndarray]:
        """Resample all synced phones onto a common uniform global timeline at target_fs."""
        if t_end_s <= t_start_s:
            return {}
        n_out = int(round((t_end_s - t_start_s) * target_fs))
        if n_out < 2:
            return {}

        t_grid = t_start_s + np.arange(n_out, dtype=np.float64) / float(target_fs)
        aligned: dict[str, np.ndarray] = {}
        for pid, slc in phone_buffers.items():
            model = self._models.get(pid)
            if model is None or not model.synced or len(slc.samples_f32) < 4:
                continue
            n_local = (t_grid - model.a_s) / model.b_s - float(slc.start_sample_index)
            aligned[pid] = map_coordinates(
                slc.samples_f32.astype(np.float64),
                [n_local],
                order=3,
                mode="constant",
                cval=0.0,
            ).astype(np.float32)
        return aligned

    def calibrate_from_buffers(
        self,
        phone_buffers: dict[str, PhoneAudioSlice],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        mode: CalibrationMode = "lfm_dual",
    ) -> CalibrationResult:
        """Calibrate clock offsets (STO) and sample-rate drift (SRO) using acoustic chirps."""
        c = speed_of_sound(room.temperature_c)
        now_ts = time.time()

        detected: dict[str, tuple[list[float], float, float, PhoneAudioSlice]] = {}
        per_phone: dict[str, PhoneSyncReport] = {}

        requires_dual = mode in ("lfm_dual", "ggwave_hybrid")

        for pid, pcfg in phones.items():
            slc = phone_buffers.get(pid)
            if slc is None or len(slc.samples_f32) == 0:
                self.mark_clock_jump(pid)
                per_phone[pid] = PhoneSyncReport(synced=False, error="No audio slice available; please recalibrate")
                continue

            arrivals, snr_db = detect_chirp_arrivals(slc.samples_f32, fs=slc.sample_rate, mode=mode)
            if not arrivals:
                self.mark_clock_jump(pid)
                per_phone[pid] = PhoneSyncReport(
                    synced=False,
                    chirp_snr_db=round(snr_db, 2),
                    error=f"Chirp not detected (SNR={snr_db:.1f} dB); please recalibrate",
                )
                continue

            if requires_dual and len(arrivals) < 2:
                self.mark_clock_jump(pid)
                per_phone[pid] = PhoneSyncReport(
                    synced=False,
                    chirp_snr_db=round(snr_db, 2),
                    error=f"Only {len(arrivals)} of 2 calibration chirps detected (SNR={snr_db:.1f} dB); please recalibrate",
                )
                continue

            tau_prop = math.hypot(pcfg.x_m - room.speaker_x_m, pcfg.y_m - room.speaker_y_m) / c
            abs_arrivals = [float(slc.start_sample_index) + n_rel for n_rel in arrivals]
            detected[pid] = (abs_arrivals, tau_prop, snr_db, slc)

        # Keep continuous global clock epoch across re-calibrations if previously synced phones exist
        anchor_t0 = 0.0
        prior_t0s: list[float] = []
        for pid, (abs_arrivals, tau_prop, _, _) in detected.items():
            prev = self._models.get(pid)
            if prev is not None and prev.synced and prev.cal_points:
                prior_t0s.append(prev.a_s + prev.b_s * abs_arrivals[0] - tau_prop)
        if prior_t0s:
            anchor_t0 = float(np.median(prior_t0s))

        # Fit linear clock model per phone
        for pid, (abs_arrivals, tau_prop, snr_db, slc) in detected.items():
            fs_nom = slc.sample_rate
            prev = self._models.get(pid)
            points = list(prev.cal_points) if (prev is not None and prev.synced) else []

            if len(abs_arrivals) >= 2:
                n0, n1 = abs_arrivals[0], abs_arrivals[1]
                points.append((n0, anchor_t0 + tau_prop))
                points.append((n1, anchor_t0 + DUAL_CHIRP_GAP_S + tau_prop))
            else:
                points.append((abs_arrivals[0], anchor_t0 + tau_prop))

            if len(points) >= 2:
                ns = np.array([pt[0] for pt in points], dtype=np.float64)
                ts = np.array([pt[1] for pt in points], dtype=np.float64)
                span_t = float(np.max(ts) - np.min(ts))
                if span_t > 0.5:
                    n_mean = float(np.mean(ns))
                    t_mean = float(np.mean(ts))
                    dn = ns - n_mean
                    denom = float(np.dot(dn, dn))
                    b_s = float(np.dot(dn, ts - t_mean) / denom) if denom > 0.0 else (1.0 / fs_nom)
                    f_hat = 1.0 / b_s if b_s > 0.0 else float(fs_nom)
                    drift_ppm = ((f_hat - fs_nom) / float(fs_nom)) * 1e6
                    if abs(drift_ppm) > 500.0:
                        # Reject physically impossible oscillator drift; fall back to nominal fs
                        b_s = 1.0 / float(fs_nom)
                        drift_ppm = 0.0
                    a_s = t_mean - b_s * n_mean
                else:
                    b_s = 1.0 / float(fs_nom)
                    a_s = ts[-1] - b_s * ns[-1]
                    drift_ppm = 0.0
            else:
                b_s = 1.0 / float(fs_nom)
                a_s = points[0][1] - b_s * points[0][0]
                drift_ppm = 0.0

            self._models[pid] = _ClockModel(
                a_s=a_s,
                b_s=b_s,
                fs_nom=fs_nom,
                drift_ppm=drift_ppm,
                chirp_snr_db=snr_db,
                last_sync_ts=now_ts,
                synced=True,
                cal_points=points[-12:],
            )

        # Honest pairwise cross-verification using aligned Chirp 1 waveforms
        synced_pids = list(detected.keys())
        residuals_ms: dict[str, float] = {pid: 0.0 for pid in synced_pids}
        if len(synced_pids) >= 2:
            max_tau_prop = max(detected[pid][1] for pid in synced_pids)
            win_start = anchor_t0 - 0.015
            win_end = anchor_t0 + max_tau_prop + CHIRP_DURATION_S + 0.025
            cal_slices = {pid: detected[pid][3] for pid in synced_pids}
            aligned_cal = self.extract_aligned_window(cal_slices, win_start, win_end, target_fs=48000)

            pair_errs: dict[str, list[float]] = {pid: [] for pid in synced_pids}
            for idx_a in range(len(synced_pids)):
                for idx_b in range(idx_a + 1, len(synced_pids)):
                    pa, pb = synced_pids[idx_a], synced_pids[idx_b]
                    if pa not in aligned_cal or pb not in aligned_cal:
                        continue
                    expected_tdoa = detected[pa][1] - detected[pb][1]
                    max_tau = abs(expected_tdoa) + 0.015
                    _, _, measured_tdoa, _ = compute_gcc_phat(
                        aligned_cal[pa],
                        aligned_cal[pb],
                        fs=48000,
                        max_tau_s=max_tau,
                        f_min=1200.0,
                        f_max=8000.0,
                    )
                    err_ms = abs(measured_tdoa - expected_tdoa) * 1000.0
                    pair_errs[pa].append(err_ms)
                    pair_errs[pb].append(err_ms)

            for pid in synced_pids:
                residuals_ms[pid] = float(np.median(pair_errs[pid])) if pair_errs[pid] else 0.0

        for pid in synced_pids:
            model = self._models[pid]
            model.residual_ms = residuals_ms[pid]
            per_phone[pid] = PhoneSyncReport(
                synced=True,
                residual_ms=round(model.residual_ms, 4),
                drift_ppm=round(model.drift_ppm, 2),
                chirp_snr_db=round(model.chirp_snr_db, 2),
            )

        all_res = [residuals_ms[pid] for pid in synced_pids]
        return CalibrationResult(
            mode=mode,
            success=len(synced_pids) >= 2,
            synced_count=len(synced_pids),
            median_residual_ms=round(float(np.median(all_res)), 4) if all_res else None,
            max_residual_ms=round(float(np.max(all_res)), 4) if all_res else None,
            per_phone=per_phone,
        )
