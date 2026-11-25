"""Cheap per-phone features for drone / clap / other (pure numpy, a few ms per window)."""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields

import numpy as np

F0_MIN, F0_MAX, N_HARMONICS = 70.0, 350.0, 8
BANDS_HZ = ((0.0, 300.0), (300.0, 2000.0), (2000.0, 24000.0))


@dataclass
class Features:
    rms: float
    crest: float           # peak / rms
    active_frac: float     # share of 10 ms frames within 20 dB of the loudest frame
    comb: float            # harmonic comb score: mean harmonic magnitude / median magnitude, in dB
    f0_hz: float
    f0_stability: float    # 1 - relative spread of f0 across sub-frames (1 = perfectly stable)
    flatness: float        # spectral flatness 0 (tonal) .. 1 (noise)
    centroid_hz: float
    concentration: float   # share of energy in the loudest 40 ms (impulsive sounds ~1)
    low_ratio: float = 0.0   # energy share below 300 Hz
    mid_ratio: float = 0.0   # 300 Hz - 2 kHz
    high_ratio: float = 0.0  # above 2 kHz
    duration_s: float = 0.0  # window length (the server picks 80 ms for transients, 750 ms for sustained)


FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Features))
# rms is a level, not a timbre: the trained model must not learn "loud = drone".
MODEL_FEATURES: tuple[str, ...] = tuple(n for n in FEATURE_NAMES if n != "rms")


def _comb(x: np.ndarray, fs: int) -> tuple[float, float]:
    n = max(len(x), 8192)
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    df = freqs[1]
    floor = float(np.median(mag[(freqs > 50) & (freqs < 3000)])) + 1e-12
    f0s = np.arange(F0_MIN, F0_MAX, 1.0)
    h = np.arange(1, N_HARMONICS + 1)
    idx = np.rint(np.outer(f0s, h) / df).astype(int)
    idx = np.clip(idx, 0, len(mag) - 2)
    peaks = np.maximum(mag[idx], mag[idx + 1])
    scores = peaks.mean(axis=1)
    best = int(np.argmax(scores))
    return 20 * np.log10(scores[best] / floor), float(f0s[best])


def extract(x: np.ndarray, fs: int) -> Features:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x))) + 1e-12
    hop = max(1, int(0.01 * fs))
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    e = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    active = float(np.mean(e > e.max() * 0.1))
    k = min(len(e), 4)
    energy = e ** 2
    conc = float(np.max(np.convolve(energy, np.ones(k), "valid")) / energy.sum())
    comb, f0 = _comb(x, fs)
    subs = [s for s in np.array_split(x, 3) if len(s) >= int(0.15 * fs)]
    if len(subs) == 3:
        f0s = np.array([_comb(s, fs)[1] for s in subs])
        stability = float(max(0.0, 1 - np.std(f0s) / max(np.mean(f0s), 1e-9) * 10))
    else:
        stability = 0.0
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    flat = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    power = mag ** 2
    total = float(power.sum())
    bands = [float(power[(freqs >= lo) & (freqs < hi)].sum()) / total for lo, hi in BANDS_HZ]
    return Features(rms, float(np.max(np.abs(x)) / rms), active, comb, f0, stability, flat, centroid, conc,
                    *bands, len(x) / fs)


def vector(f: Features, names: tuple[str, ...] = MODEL_FEATURES) -> np.ndarray:
    d = dict(zip(FEATURE_NAMES, astuple(f)))
    return np.array([d[n] for n in names], dtype=np.float64)
