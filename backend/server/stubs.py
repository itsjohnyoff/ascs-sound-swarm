"""Fallback facades with the exact CONTRACTS 7 signatures.

The session uses these when the real DSP or classifier module is missing, so the
server always starts. They never sync, never localize, and always say "other".
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from shared.protocol import (
    CalibrationMode,
    CalibrationResult,
    ClassificationResult,
    LocalizationResult,
    PhoneAudioSlice,
    PhoneConfig,
    PhoneSyncReport,
    RoomConfig,
)


class StubSyncEngine:
    def add_chirp_observation(self, phone_buffers: dict[str, PhoneAudioSlice], phones: dict[str, PhoneConfig],
                              room: RoomConfig, mode: CalibrationMode) -> CalibrationResult:
        return CalibrationResult(
            mode=mode, success=False, synced_count=0,
            per_phone={pid: PhoneSyncReport(synced=False, error="DSP module unavailable") for pid in phones},
        )

    def clock(self, phone_id: str):
        return None

    def is_synced(self, phone_id: str) -> bool:
        return False

    def to_global(self, phone_id: str, sample_index: float) -> float | None:
        return None

    def to_local(self, phone_id: str, t_global: float) -> float | None:
        return None

    def mark_clock_jump(self, phone_id: str) -> None:
        pass

    def extract_aligned_window(self, phone_buffers: dict[str, PhoneAudioSlice], t_start: float, t_end: float,
                               target_fs: int = 48000) -> dict[str, np.ndarray]:
        return {}


class StubOnsetDetector:
    def push(self, samples: np.ndarray, start_sample_index: int, fs: int) -> list[int]:
        return []


class StubLocalizationPipeline:
    def localize(self, aligned: dict[str, np.ndarray], phones: dict[str, PhoneConfig], room: RoomConfig,
                 kind: Literal["transient", "sustained"], fs: int = 48000) -> LocalizationResult | None:
        return None


class StubSoundClassifier:
    backend_name = "stub"

    def classify_swarm(self, aligned: dict[str, np.ndarray], fs: int = 48000) -> ClassificationResult:
        return ClassificationResult(
            sound_class="other", is_drone=False, confidence=0.0,
            probs={"drone": 0.0, "clap": 0.0, "other": 1.0}, backend_name=self.backend_name,
        )
