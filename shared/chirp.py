"""The one LFM calibration chirp implementation (CONTRACTS 1). Golden vectors: shared/chirp_<fs>.f32."""

from __future__ import annotations

import math

import numpy as np

from shared.protocol import (
    CHIRP_AMPLITUDE,
    CHIRP_DURATION_S,
    CHIRP_F0_HZ,
    CHIRP_F1_HZ,
    CHIRP_TUKEY_ALPHA,
)


def generate_lfm_chirp(
    fs: int,
    f0: float = CHIRP_F0_HZ,
    f1: float = CHIRP_F1_HZ,
    duration_s: float = CHIRP_DURATION_S,
    tukey_alpha: float = CHIRP_TUKEY_ALPHA,
    amplitude: float = CHIRP_AMPLITUDE,
) -> np.ndarray:
    """Generate the exact deterministic LFM up-chirp defined in shared contracts."""
    n = int(round(fs * duration_s))
    if n < 2:
        raise ValueError(f"Chirp sample length too short: {n}")

    k = np.arange(n, dtype=np.float64)
    t = k / float(fs)
    u = k / float(n - 1)

    w = np.ones(n, dtype=np.float64)
    if tukey_alpha > 0.0:
        half = 0.5 * tukey_alpha
        left = u < half
        right = u > (1.0 - half)
        w[left] = 0.5 * (1.0 - np.cos(2.0 * math.pi * u[left] / tukey_alpha))
        w[right] = 0.5 * (1.0 - np.cos(2.0 * math.pi * (1.0 - u[right]) / tukey_alpha))

    chirp_rate = (f1 - f0) / duration_s
    phase = 2.0 * math.pi * (f0 * t + 0.5 * chirp_rate * t * t)
    return (amplitude * w * np.sin(phase)).astype(np.float32)
