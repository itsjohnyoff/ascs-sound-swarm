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


# ---------------------------------------------------------------------------
# Extra sources for classifier training (not used by the scenarios above, so their
# random streams stay unchanged).
# ---------------------------------------------------------------------------
def multirotor_drone(rng: np.random.Generator, duration_s: float = 0.9) -> np.ndarray:
    """2-4 rotors with slightly different blade-pass f0 (beating), uneven harmonics, motor whine."""
    n = int(duration_s * SRC_FS)
    t = np.arange(n) / SRC_FS
    base = float(rng.uniform(80, 300))
    x = np.zeros(n)
    for _ in range(int(rng.integers(2, 5))):
        f0 = base * (1 + rng.uniform(-0.03, 0.03))
        wobble = f0 * (1 + rng.uniform(0.002, 0.03) * np.sin(2 * np.pi * rng.uniform(0.2, 3.0) * t
                                                               + rng.uniform(0, 2 * np.pi)))
        phase = 2 * np.pi * np.cumsum(wobble) / SRC_FS
        amps = rng.uniform(0.2, 1.0, 10) / np.arange(1, 11) ** rng.uniform(0.3, 1.2)
        x += sum(a * np.sin(h * phase + rng.uniform(0, 2 * np.pi)) for h, a in enumerate(amps, 1))
    whine = rng.uniform(0, 0.15) * np.sin(2 * np.pi * rng.uniform(1500, 4000) * t)
    sos = butter(2, [100, 8000], btype="band", fs=SRC_FS, output="sos")
    x = x / np.max(np.abs(x)) + whine + rng.uniform(0.1, 0.6) * sosfilt(sos, rng.standard_normal(n))
    return 0.6 * x / np.max(np.abs(x))


def knock(rng: np.random.Generator, duration_s: float = 0.15) -> np.ndarray:
    """Door knock / footstep / table tap: low, damped resonances, a little noise."""
    n = int(duration_s * SRC_FS)
    t = np.arange(n) / SRC_FS
    x = sum(np.sin(2 * np.pi * rng.uniform(80, 900) * t + rng.uniform(0, 6.3)) * np.exp(-t / rng.uniform(0.008, 0.04))
            for _ in range(3))
    x = x + 0.2 * rng.standard_normal(n) * np.exp(-t / 0.004)
    return 0.8 * x / np.max(np.abs(x))


def noise_burst(rng: np.random.Generator, duration_s: float = 1.0) -> np.ndarray:
    """Fan, HVAC, rustling, crowd murmur: coloured noise with a slow envelope."""
    n = int(duration_s * SRC_FS)
    lo = float(rng.uniform(60, 800))
    hi = float(min(lo * rng.uniform(3, 30), 20000))
    sos = butter(2, [lo, hi], btype="band", fs=SRC_FS, output="sos")
    t = np.arange(n) / SRC_FS
    env = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 4) * t) * rng.uniform(0, 1)
    x = sosfilt(sos, rng.standard_normal(n)) * env
    return 0.5 * x / np.max(np.abs(x))


def whistle(rng: np.random.Generator, duration_s: float = 1.0) -> np.ndarray:
    """Whistle / beep / alarm: one high tone with vibrato, maybe one overtone."""
    n = int(duration_s * SRC_FS)
    t = np.arange(n) / SRC_FS
    f = float(rng.uniform(600, 3500)) * (1 + 0.01 * np.sin(2 * np.pi * rng.uniform(3, 7) * t))
    phase = 2 * np.pi * np.cumsum(f) / SRC_FS
    x = np.sin(phase) + rng.uniform(0, 0.3) * np.sin(2 * phase)
    x = x * np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02) + 0.02 * rng.standard_normal(n)
    return 0.5 * x / np.max(np.abs(x))


def music_like(rng: np.random.Generator, duration_s: float = 1.0) -> np.ndarray:
    """Short melody: 3-6 notes with harmonics that change pitch every 100-300 ms."""
    n = int(duration_s * SRC_FS)
    x = np.zeros(n)
    pos = 0
    while pos < n:
        m = min(n - pos, int(rng.uniform(0.1, 0.3) * SRC_FS))
        t = np.arange(m) / SRC_FS
        f0 = 110 * 2 ** (rng.integers(0, 30) / 12)
        note = sum(np.sin(2 * np.pi * h * f0 * t) / h ** 1.5 for h in range(1, 6))
        x[pos:pos + m] = note * np.exp(-t / rng.uniform(0.05, 0.3))
        pos += m
    return 0.5 * x / np.max(np.abs(x))


TRAINING_SOURCES = {
    "drone": (drone, multirotor_drone),
    "clap": (clap,),
    "other": (speech_like, knock, noise_burst, whistle, music_like),
}


def render(kind: str, rng: np.random.Generator, duration_s: float | None) -> np.ndarray:
    if kind == "chirp":
        return chirp()
    kw = {} if duration_s is None else {"duration_s": duration_s}
    return {"clap": clap, "drone": drone, "speech": speech_like}[kind](rng, **kw)
