"""Deterministic LFM calibration chirp synthesis and optional ggwave hybrid/benchmark."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from shared.protocol import (
    CHIRP_AMPLITUDE,
    CHIRP_DURATION_S,
    CHIRP_F0_HZ,
    CHIRP_F1_HZ,
    CHIRP_TUKEY_ALPHA,
    DUAL_CHIRP_GAP_S,
)

try:
    import ggwave as _ggwave  # type: ignore[import-not-found]

    GGWAVE_AVAILABLE = True
except ImportError:
    _ggwave = None
    GGWAVE_AVAILABLE = False


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


def generate_dual_chirp(
    fs: int,
    gap_s: float = DUAL_CHIRP_GAP_S,
    f0: float = CHIRP_F0_HZ,
    f1: float = CHIRP_F1_HZ,
    duration_s: float = CHIRP_DURATION_S,
    tukey_alpha: float = CHIRP_TUKEY_ALPHA,
    amplitude: float = CHIRP_AMPLITUDE,
) -> np.ndarray:
    """Generate a single continuous buffer containing two chirps separated by exact gap_s."""
    single = generate_lfm_chirp(
        fs=fs,
        f0=f0,
        f1=f1,
        duration_s=duration_s,
        tukey_alpha=tukey_alpha,
        amplitude=amplitude,
    )
    gap_samples = int(round(fs * gap_s))
    buf = np.zeros(gap_samples + len(single), dtype=np.float32)
    buf[: len(single)] = single
    buf[gap_samples : gap_samples + len(single)] = single
    return buf


def encode_ggwave_preamble(payload: str = "C01", fs: int = 48000) -> np.ndarray | None:
    """Encode an acoustic calibration tag using ggwave if the C++ binding is installed."""
    if not GGWAVE_AVAILABLE or _ggwave is None:
        return None
    # shortcut: ggwave python bindings target 48 kHz mono int16 PCM; skips resampling if fs==48000
    _ = fs
    raw_bytes = _ggwave.encode(payload)
    pcm = np.frombuffer(raw_bytes, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).copy()


def decode_ggwave_preamble(audio: np.ndarray, fs: int = 48000) -> str | None:
    """Decode a ggwave calibration tag from float32 audio, or return None if unavailable/not found."""
    if not GGWAVE_AVAILABLE or _ggwave is None:
        return None
    _ = fs
    pcm_bytes = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    instance = _ggwave.init()
    try:
        decoded = _ggwave.decode(instance, pcm_bytes)
        if decoded:
            return decoded.decode("utf-8", errors="ignore")
        return None
    finally:
        if hasattr(_ggwave, "free"):
            _ggwave.free(instance)


def generate_hybrid_chirp(
    fs: int,
    payload: str = "C01",
    gap_s: float = DUAL_CHIRP_GAP_S,
) -> np.ndarray:
    """Prepend optional ggwave calibration ID before the dual LFM timing chirp."""
    dual = generate_dual_chirp(fs=fs, gap_s=gap_s)
    preamble = encode_ggwave_preamble(payload=payload, fs=fs)
    if preamble is None:
        return dual
    pad = np.zeros(int(round(0.10 * fs)), dtype=np.float32)
    return np.concatenate([preamble, pad, dual])


def benchmark_ggwave_vs_lfm(
    audio_clips: Sequence[np.ndarray],
    true_delays_s: Sequence[float],
    fs: int = 48000,
) -> dict[str, float | bool | str]:
    """Quantify LFM matched-filter sub-sample accuracy vs ggwave frame granularity."""
    from backend.dsp.sync import detect_chirp_arrivals

    lfm_errors_ms: list[float] = []
    ggwave_decoded = 0
    for clip, true_d in zip(audio_clips, true_delays_s):
        arrivals, _ = detect_chirp_arrivals(clip, fs=fs, mode="lfm_single")
        if arrivals:
            lfm_errors_ms.append(abs(arrivals[0] / fs - true_d) * 1000.0)
        if GGWAVE_AVAILABLE and decode_ggwave_preamble(clip, fs=fs) is not None:
            ggwave_decoded += 1

    # ggwave operates on 1024-sample FSK FFT frames (~21.33 ms at 48 kHz) without sub-sample TOA
    ggwave_frame_granularity_ms = (1024.0 / fs) * 1000.0
    return {
        "ggwave_available": GGWAVE_AVAILABLE,
        "clips_evaluated": float(len(audio_clips)),
        "lfm_detected": float(len(lfm_errors_ms)),
        "lfm_mae_ms": float(np.mean(lfm_errors_ms)) if lfm_errors_ms else float("inf"),
        "lfm_max_ms": float(np.max(lfm_errors_ms)) if lfm_errors_ms else float("inf"),
        "ggwave_decoded_count": float(ggwave_decoded),
        "ggwave_fsk_frame_granularity_ms": ggwave_frame_granularity_ms,
        "summary": (
            f"LFM MAE={np.mean(lfm_errors_ms):.4f} ms vs ggwave FSK frame step {ggwave_frame_granularity_ms:.2f} ms"
            if lfm_errors_ms
            else "No LFM detections"
        ),
    }
