"""Source waveforms at 48 kHz, rendered later at any delay by sinc interpolation."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from shared.chirp import generate_lfm_chirp

SRC_FS = 48000


def chirp() -> np.ndarray:
    return generate_lfm_chirp(SRC_FS).astype(np.float64)


def clap(rng: np.random.Generator, duration_s: float = 0.12) -> np.ndarray:
    n = int(duration_s * SRC_FS)
    sos = butter(4, [400, 6000], btype="band", fs=SRC_FS, output="sos")
    noise = sosfilt(sos, rng.standard_normal(n))
    t = np.arange(n) / SRC_FS
    env = np.exp(-t / 0.020) * np.minimum(1.0, t / 0.0008)
    x = noise * env
    return 0.9 * x / np.max(np.abs(x))


def drone(rng: np.random.Generator, duration_s: float = 0.9, f0: float | None = None) -> np.ndarray:
    n = int(duration_s * SRC_FS)
    t = np.arange(n) / SRC_FS
    f0 = float(rng.uniform(80, 240)) if f0 is None else f0
    wobble = f0 * (1 + 0.02 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t))
    phase = 2 * np.pi * np.cumsum(wobble) / SRC_FS
    x = sum(np.sin(h * phase + rng.uniform(0, 2 * np.pi)) / h for h in range(1, 9))
    sos = butter(2, [150, 6000], btype="band", fs=SRC_FS, output="sos")
    x = x + 0.3 * sosfilt(sos, rng.standard_normal(n))
    ramp = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
    x = x * ramp
    return 0.6 * x / np.max(np.abs(x))


def speech_like(rng: np.random.Generator, duration_s: float = 1.0) -> np.ndarray:
    n = int(duration_s * SRC_FS)
    t = np.arange(n) / SRC_FS
    sos = butter(2, [300, 3000], btype="band", fs=SRC_FS, output="sos")
    env = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 2
    x = sosfilt(sos, rng.standard_normal(n)) * env
    return 0.5 * x / np.max(np.abs(x))


def render(kind: str, rng: np.random.Generator, duration_s: float | None) -> np.ndarray:
    if kind == "chirp":
        return chirp()
    kw = {} if duration_s is None else {"duration_s": duration_s}
    return {"clap": clap, "drone": drone, "speech": speech_like}[kind](rng, **kw)
