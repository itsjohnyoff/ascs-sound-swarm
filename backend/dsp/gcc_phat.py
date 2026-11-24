"""Band-limited GCC-PHAT-beta delay estimation with physical bounds & PSR weighting."""

from __future__ import annotations

import math

import numpy as np


def _parabolic_peak_offset(y_m1: float, y_0: float, y_p1: float) -> float:
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-18:
        return 0.0
    delta = 0.5 * (y_m1 - y_p1) / denom
    return float(np.clip(delta, -0.5, 0.5))


def compute_gcc_phat(
    sig_i: np.ndarray,
    sig_j: np.ndarray,
    fs: int,
    max_tau_s: float,
    f_min: float = 300.0,
    f_max: float = 6000.0,
    phat_beta: float = 0.8,
    interp_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute band-limited GCC-PHAT-beta between sig_i and sig_j.

    Sign convention: tau_ij = t_i - t_j. Positive tau means sig_i arrives AFTER sig_j.
    Returns (lags_s, corr_curve, best_tau_s, psr_weight).
    """
    n_i = len(sig_i)
    n_j = len(sig_j)
    if n_i < 8 or n_j < 8 or max_tau_s <= 0.0:
        lags = np.array([0.0], dtype=np.float64)
        corr = np.array([0.0], dtype=np.float64)
        return lags, corr, 0.0, 0.0

    n_lin = n_i + n_j - 1
    n_fft = 1 << (n_lin - 1).bit_length()

    xi = np.fft.rfft(sig_i.astype(np.float64), n=n_fft)
    xj = np.fft.rfft(sig_j.astype(np.float64), n=n_fft)
    cross = xi * np.conj(xj)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    hi_cut = min(f_max, 0.48 * fs)
    band_mask = (freqs >= f_min) & (freqs <= hi_cut)

    mag = np.abs(cross)
    max_mag = float(np.max(mag[band_mask])) if np.any(band_mask) else 0.0
    if max_mag < 1e-18:
        lags = np.array([0.0], dtype=np.float64)
        corr = np.array([0.0], dtype=np.float64)
        return lags, corr, 0.0, 0.0

    denom = np.power(mag, phat_beta) + 1e-8 * (math.pow(max_mag, phat_beta) + 1e-12)
    weighted = np.where(band_mask, cross / denom, 0.0 + 0.0j)

    n_out = n_fft * interp_factor
    corr_full = np.fft.irfft(weighted, n=n_out) * interp_factor

    fs_interp = float(fs * interp_factor)
    max_lag_idx = min(int(math.ceil(max_tau_s * fs_interp)), n_out // 2 - 1)
    if max_lag_idx < 1:
        lags = np.array([0.0], dtype=np.float64)
        corr = np.array([float(corr_full[0])], dtype=np.float64)
        return lags, corr, 0.0, 0.05

    corr_slice = np.concatenate((corr_full[-max_lag_idx:], corr_full[: max_lag_idx + 1]))
    lags_s = np.arange(-max_lag_idx, max_lag_idx + 1, dtype=np.float64) / fs_interp

    # Strictly enforce physical distance bound |tau| <= max_tau_s
    valid = np.abs(lags_s) <= max_tau_s
    lags_s = lags_s[valid]
    corr_slice = corr_slice[valid]

    k_max = int(np.argmax(corr_slice))
    if 0 < k_max < len(corr_slice) - 1:
        delta = _parabolic_peak_offset(
            float(corr_slice[k_max - 1]),
            float(corr_slice[k_max]),
            float(corr_slice[k_max + 1]),
        )
    else:
        delta = 0.0
    best_tau_s = float(lags_s[k_max] + delta / fs_interp)
    best_tau_s = float(np.clip(best_tau_s, -max_tau_s, max_tau_s))

    # Peak-to-Sidelobe Ratio excluding +-0.5 ms around peak
    excl = max(1, int(round(0.0005 * fs_interp)))
    mask_side = np.ones(len(corr_slice), dtype=bool)
    mask_side[max(0, k_max - excl) : min(len(corr_slice), k_max + excl + 1)] = False
    if np.any(mask_side):
        side = corr_slice[mask_side]
        psr = (float(corr_slice[k_max]) - float(np.mean(side))) / (float(np.std(side)) + 1e-12)
    else:
        psr = 3.0
    weight = float(np.clip((psr - 2.0) / 10.0, 0.05, 1.0))
    return lags_s, corr_slice, best_tau_s, weight
