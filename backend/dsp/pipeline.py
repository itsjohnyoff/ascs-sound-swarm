"""Public DSP facades matching shared/protocol.py, 02_SHARED_CONTRACTS.md, and CONTRACTS.md v2."""

from __future__ import annotations

from typing import Literal

import numpy as np

from shared.protocol import LocalizationResult, PhoneConfig, RoomConfig
from backend.dsp.clock import ClockModel
from backend.dsp.localizer import AcousticLocalizer2D
from backend.dsp.sync import SyncEngine
from backend.dsp.trigger import AdaptiveEventTrigger, EventWindow, OnsetDetector


class LocalizationPipeline:
    """End-to-end event trigger + 2D SRP-PHAT + Huber multilateration pipeline."""

    def __init__(self, grid_spacing_m: float = 0.15) -> None:
        self._trigger = AdaptiveEventTrigger()
        self._localizer = AcousticLocalizer2D(grid_spacing_m=grid_spacing_m)
        self._event_counter = 0

    def localize(
        self,
        aligned: dict[str, np.ndarray],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        kind: Literal["transient", "sustained"],
        fs: int = 48000,
        timestamp_s: float = 0.0,
    ) -> LocalizationResult | None:
        """v2 localization entry point for server-selected event windows (no re-triggering)."""
        eligible = {
            pid: np.asarray(arr, dtype=np.float32)
            for pid, arr in aligned.items()
            if pid in phones and len(arr) >= 8 and bool(np.all(np.isfinite(arr)))
        }
        if len(eligible) < 2:
            return None

        n_samples = min(len(arr) for arr in eligible.values())
        if n_samples < 8:
            return None

        if kind == "transient":
            frames = [{pid: arr[:n_samples] for pid, arr in eligible.items()}]
        else:
            frame_len = min(n_samples, int(round(0.250 * fs)))
            step = max(1, frame_len // 2)
            hann = np.hanning(frame_len).astype(np.float32)
            frames = []
            pos = 0
            while pos + frame_len <= n_samples and len(frames) < 3:
                frames.append({pid: arr[pos : pos + frame_len] * hann for pid, arr in eligible.items()})
                pos += step
            if not frames:
                w = np.hanning(n_samples).astype(np.float32)
                frames.append({pid: arr[:n_samples] * w for pid, arr in eligible.items()})

        event = EventWindow(kind=kind, frames=frames, onset_offset_s=0.0, peak_snr_db=20.0)
        self._event_counter += 1
        event_id = f"evt-{self._event_counter:04d}"
        return self._localizer.localize_event(
            event=event,
            phones=phones,
            room=room,
            fs=fs,
            event_id=event_id,
            timestamp_s=timestamp_s,
        )

    def detect_and_localize(
        self,
        aligned_audio: dict[str, np.ndarray],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        fs: int = 48000,
        timestamp_s: float = 0.0,
    ) -> LocalizationResult | None:
        """Legacy v1 entry point preserved during migration."""
        event = self._trigger.detect(aligned_audio, fs=fs)
        if event is None:
            return None
        self._event_counter += 1
        event_id = f"evt-{self._event_counter:04d}"
        return self._localizer.localize_event(
            event=event,
            phones=phones,
            room=room,
            fs=fs,
            event_id=event_id,
            timestamp_s=timestamp_s,
        )


__all__ = ["ClockModel", "SyncEngine", "OnsetDetector", "LocalizationPipeline"]
