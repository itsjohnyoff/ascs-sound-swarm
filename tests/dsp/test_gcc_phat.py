"""TDD tests for band-limited GCC-PHAT-beta and adaptive event trigger."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from backend.dsp.gcc_phat import compute_gcc_phat
from backend.dsp.trigger import AdaptiveEventTrigger


def test_gcc_phat_recovers_fractional_delay_and_sign_convention() -> None:
    fs = 48000
    rng = np.random.default_rng(123)
    duration_s = 0.08
    n = int(round(duration_s * fs))

    # Broadband clap-like impulse signal
    src = np.zeros(n + 2000, dtype=np.float64)
    burst = rng.normal(0.0, 1.0, size=int(0.025 * fs)) * np.hanning(int(0.025 * fs))
    src[1000 : 1000 + len(burst)] = burst

    # sig_i arrives +3.72 ms AFTER sig_j => tau_ij = t_i - t_j = +0.00372 s
    true_tau_s = 0.00372
    idx_j = np.arange(n, dtype=np.float64) + 800.0
    idx_i = idx_j - true_tau_s * fs
    sig_j = map_coordinates(src, [idx_j], order=3, mode="constant", cval=0.0).astype(np.float32)
    sig_i = map_coordinates(src, [idx_i], order=3, mode="constant", cval=0.0).astype(np.float32)

    # Add 10 dB independent noise
    sig_std = float(np.std(sig_j))
    noise_std = sig_std / (10.0 ** (10.0 / 20.0))
    sig_j_noisy = (sig_j + rng.normal(0.0, noise_std, size=n)).astype(np.float32)
    sig_i_noisy = (sig_i + rng.normal(0.0, noise_std, size=n)).astype(np.float32)

    lags, corr, est_tau_s, weight = compute_gcc_phat(sig_i_noisy, sig_j_noisy, fs=fs, max_tau_s=0.015)
    err_ms = abs(est_tau_s - true_tau_s) * 1000.0
    print(f"\nGCC-PHAT fractional delay error (10 dB noise): {err_ms:.4f} ms (weight={weight:.2f})")
    assert err_ms < 0.05
    assert weight > 0.3
    assert len(lags) == len(corr)


def test_gcc_phat_rejects_impossible_out_of_bounds_peak() -> None:
    fs = 48000
    rng = np.random.default_rng(9)
    n = int(0.10 * fs)
    base = rng.normal(0.0, 1.0, size=n).astype(np.float32)

    # Dominant bogus copy shifted by +25 ms (impossible when max_tau_s = 8 ms),
    # plus valid direct signal shifted by -2.5 ms.
    sig_j = base.copy()
    sig_i = np.roll(base, int(round(0.025 * fs))) * 1.5 + np.roll(base, -int(round(0.0025 * fs))) * 0.7

    _, _, est_tau_s, _ = compute_gcc_phat(sig_i, sig_j, fs=fs, max_tau_s=0.008)
    assert abs(est_tau_s - (-0.0025)) < 0.0002


def test_adaptive_trigger_distinguishes_clap_transient_vs_sustained_drone() -> None:
    fs = 48000
    rng = np.random.default_rng(77)
    trigger = AdaptiveEventTrigger()

    # 1. Quiet noise should not trigger
    quiet = {f"p{i}": rng.normal(0.0, 1e-4, size=int(0.5 * fs)).astype(np.float32) for i in range(4)}
    assert trigger.detect(quiet, fs) is None

    # 2. Sharp clap transient at t = 0.20 s
    clap_win = {f"p{i}": rng.normal(0.0, 1e-4, size=int(0.5 * fs)).astype(np.float32) for i in range(4)}
    t_burst = np.arange(int(0.020 * fs)) / fs
    clap_pulse = (0.4 * np.exp(-t_burst / 0.005) * rng.normal(0.0, 1.0, size=len(t_burst))).astype(np.float32)
    for i in range(4):
        s = int((0.20 + i * 0.002) * fs)
        clap_win[f"p{i}"][s : s + len(clap_pulse)] += clap_pulse
    ev_clap = trigger.detect(clap_win, fs)
    assert ev_clap is not None
    assert ev_clap.kind == "transient"
    assert len(ev_clap.frames) == 1
    assert 0.06 <= len(ev_clap.frames[0]["p0"]) / fs <= 0.085

    # 3. Sustained continuous drone buzz across full 0.5 s window
    t = np.arange(int(0.5 * fs)) / fs
    drone_sig = sum(0.04 / k * np.sin(2.0 * np.pi * 160.0 * k * t) for k in range(1, 9)).astype(np.float32)
    drone_win = {f"p{i}": (drone_sig + rng.normal(0.0, 2e-4, size=len(t))).astype(np.float32) for i in range(4)}
    ev_drone = trigger.detect(drone_win, fs)
    assert ev_drone is not None
    assert ev_drone.kind == "sustained"
    assert len(ev_drone.frames) >= 2
