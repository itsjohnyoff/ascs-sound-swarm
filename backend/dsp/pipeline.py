"""Public DSP facades matching shared/protocol.py and 02_SHARED_CONTRACTS.md."""

from __future__ import annotations

import numpy as np

from shared.protocol import LocalizationResult, PhoneConfig, RoomConfig
from backend.dsp.localizer import AcousticLocalizer2D
from backend.dsp.sync import SyncEngine
from backend.dsp.trigger import AdaptiveEventTrigger


class LocalizationPipeline:
    """End-to-end event trigger + 2D SRP-PHAT + Huber multilateration pipeline."""

    def __init__(self, grid_spacing_m: float = 0.15) -> None:
        self._trigger = AdaptiveEventTrigger()
        self._localizer = AcousticLocalizer2D(grid_spacing_m=grid_spacing_m)
        self._event_counter = 0

    def detect_and_localize(
        self,
        aligned_audio: dict[str, np.ndarray],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        fs: int = 48000,
        timestamp_s: float = 0.0,
    ) -> LocalizationResult | None:
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


__all__ = ["SyncEngine", "LocalizationPipeline"]
