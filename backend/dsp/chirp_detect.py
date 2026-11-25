"""Offline full-recording chirp detector wrapper (CONTRACTS §2 / Gate 0)."""

from __future__ import annotations

import numpy as np

from backend.dsp.clock import ChirpArrival
from backend.dsp.sync import detect_all_chirps

Arrival = ChirpArrival


def detect_chirps(
    pcm: np.ndarray,
    fs: int,
    template: np.ndarray | None = None,
    start_sample_index: int = 0,
    valid_mask: np.ndarray | None = None,
    min_spacing_s: float = 1.0,
    min_snr_db: float = 12.0,
    min_ncc: float = 0.25,
    lookback_s: float = 0.025,
    direct_path_ratio: float = 0.55,
    f_low: float = 1200.0,
    f_high: float = 8500.0,
) -> list[Arrival]:
    """Return every chirp's fractional sample index and SNR across a full recording."""
    return detect_all_chirps(
        samples_f32=pcm,
        fs=fs,
        template=template,
        start_sample_index=start_sample_index,
        valid_mask=valid_mask,
        min_spacing_s=min_spacing_s,
        min_snr_db=min_snr_db,
        min_ncc=min_ncc,
        lookback_s=lookback_s,
        direct_path_ratio=direct_path_ratio,
        f_low=f_low,
        f_high=f_high,
    )


__all__ = ["Arrival", "detect_chirps"]
