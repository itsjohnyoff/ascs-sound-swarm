"""Augmentation for classifier training: room impulse responses, phone-speaker playback, noise, level.

Uses the same physics as `sim.room` (direct path + first-order wall images) plus an
exponentially decaying diffuse tail for RT60 in 0.2-0.7 s, which the 4-image model lacks.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from sim.sources import SRC_FS

C = 343.0


def random_rir(rng: np.random.Generator, fs: int = SRC_FS) -> np.ndarray:
    w, h = rng.uniform(4, 12), rng.uniform(3, 9)
    sx, sy = rng.uniform(0.3, w - 0.3), rng.uniform(0.3, h - 0.3)
    mx, my = rng.uniform(0.3, w - 0.3), rng.uniform(0.3, h - 0.3)
    rt60 = rng.uniform(0.2, 0.7)
    refl = rng.uniform(0.3, 0.8)
    images = [((sx, sy), 1.0), ((-sx, sy), refl), ((2 * w - sx, sy), refl), ((sx, -sy), refl), ((sx, 2 * h - sy), refl)]
    d0 = float(np.hypot(sx - mx, sy - my))
    n = int((d0 / C + rt60) * fs) + 64
    rir = np.zeros(n)
    for (ix, iy), g in images:
        d = float(np.hypot(ix - mx, iy - my))
        k = (d - d0) / C * fs + 8
        i = int(k)
        if i + 1 < n:
            frac = k - i
            amp = g * d0 / max(d, 0.3)
            rir[i] += amp * (1 - frac)
            rir[i + 1] += amp * frac
    t = np.arange(n) / fs
    start = int(rng.uniform(0.002, 0.01) * fs) + 8
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * rng.uniform(0.05, 0.3)
    tail[:start] = 0
    return rir + tail


def phone_speaker(x: np.ndarray, rng: np.random.Generator, fs: int = SRC_FS) -> np.ndarray:
    """Judges may play drone audio from a phone: 200 Hz - 6 kHz band, slight saturation."""
    sos = butter(4, [rng.uniform(150, 350), rng.uniform(5000, 8000)], btype="band", fs=fs, output="sos")
    y = sosfilt(sos, x)
    drive = rng.uniform(1.0, 3.0)
    return np.tanh(drive * y / (np.max(np.abs(y)) + 1e-12)) / np.tanh(drive)


def augment(x: np.ndarray, rng: np.random.Generator, *, fs: int = SRC_FS, p_rir: float = 0.8,
            p_speaker: float = 0.3, snr_db: tuple[float, float] = (5.0, 35.0), p_lowpass: float = 0.3):
    """Returns (augmented signal, SNR in dB to mix room noise at).

    p_lowpass: band-limit to 5-8 kHz for every class, so 16 kHz dataset recordings (no
    energy above 8 kHz) do not teach the model "dull = drone".
    """
    y = np.asarray(x, dtype=np.float64)
    if rng.random() < p_speaker:
        y = phone_speaker(y, rng, fs)
    if rng.random() < p_lowpass:
        y = sosfilt(butter(6, rng.uniform(5000, 8000), btype="low", fs=fs, output="sos"), y)
    if rng.random() < p_rir:
        y = fftconvolve(y, random_rir(rng, fs))[: len(y) + int(0.3 * fs)]
    y = y / (np.max(np.abs(y)) + 1e-12) * rng.uniform(0.02, 0.9)
    return y, float(rng.uniform(*snr_db))


def place_in_window(x: np.ndarray, kind: str, rng: np.random.Generator, snr_db: float,
                    fs: int = SRC_FS, noise_ref: np.ndarray | None = None) -> np.ndarray:
    """Cut the window the server would hand the classifier and add room noise.

    transient: 80 ms, onset 15 ms (+-3 ms jitter) after the window start.
    sustained: 750 ms, starting 0-300 ms after the onset (later updates of a drone).
    """
    if kind == "transient":
        n, pre = int(0.08 * fs), int((0.015 + rng.uniform(-0.003, 0.003)) * fs)
        seg = np.zeros(n)
        m = min(len(x), n - pre)
        seg[pre:pre + m] = x[:m]
    else:
        n = int(0.75 * fs)
        off = int(rng.integers(0, min(len(x) - n, int(0.3 * fs)) + 1)) if len(x) > n else 0
        seg = np.zeros(n)
        part = x[off:off + n]
        seg[:len(part)] = part
    sig_rms = float(np.sqrt(np.mean(seg[np.abs(seg) > 0] ** 2))) if np.any(seg) else 1e-3
    if noise_ref is not None and len(noise_ref) >= n:
        i = int(rng.integers(0, len(noise_ref) - n + 1))
        noise = noise_ref[i:i + n] / (np.sqrt(np.mean(noise_ref[i:i + n] ** 2)) + 1e-12)
    else:
        sos = butter(1, [40, 12000], btype="band", fs=fs, output="sos")
        noise = sosfilt(sos, rng.standard_normal(n))
        noise /= np.sqrt(np.mean(noise ** 2)) + 1e-12
    return seg + noise * sig_rms * 10 ** (-snr_db / 20)
