"""Public DSP facades matching shared/protocol.py, 02_SHARED_CONTRACTS.md, and CONTRACTS.md v2."""

from __future__ import annotations

from typing import Literal

import numpy as np

from shared.protocol import LocalizationResult, PhoneConfig, RoomConfig
from backend.dsp.clock import ClockModel
from backend.dsp.localizer import AcousticLocalizer2D
from backend.dsp.sync import CLIP_THRESHOLD, SyncEngine, is_window_clipped
from backend.dsp.trigger import AdaptiveEventTrigger, EventWindow, OnsetDetector


def _filter_clipped_phones(
    windows: dict[str, np.ndarray],
    clip_threshold: float = CLIP_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Drop clipped phones when at least 3 unclipped phones remain for 2D TDOA."""
    clean = {
        pid: arr
        for pid, arr in windows.items()
        if not is_window_clipped(arr, clip_threshold=clip_threshold)
    }
    if len(clean) >= 3:
        return clean
    # shortcut: retain phones when dropping clipped ones would leave <3 phones for 2D TDOA
    return windows


class LocalizationPipeline:
    """End-to-end event trigger + 2D SRP-PHAT + Huber multilateration pipeline."""

    def __init__(
        self,
        grid_spacing_m: float = 0.15,
        *,
        f_min: float = 300.0,
        f_max: float = 6000.0,
        phat_beta: float = 0.8,
        cycle_threshold_s: float = 0.0015,
        loo_residual_threshold_s: float = 0.0005,
        sustained_frame_s: float = 0.250,
        sustained_max_frames: int = 3,
        transient_pre_s: float = 0.015,
        transient_post_s: float = 0.065,
    ) -> None:
        self.sustained_frame_s = sustained_frame_s
        self.sustained_max_frames = sustained_max_frames
        self.transient_pre_s = transient_pre_s
        self.transient_post_s = transient_post_s
        self._trigger = AdaptiveEventTrigger(
            sustained_frame_s=sustained_frame_s,
            sustained_max_frames=sustained_max_frames,
            transient_pre_s=transient_pre_s,
            transient_post_s=transient_post_s,
        )
        self._localizer = AcousticLocalizer2D(
            grid_spacing_m=grid_spacing_m,
            f_min=f_min,
            f_max=f_max,
            phat_beta=phat_beta,
            cycle_threshold_s=cycle_threshold_s,
            loo_residual_threshold_s=loo_residual_threshold_s,
        )
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

        eligible = {pid: arr[:n_samples] for pid, arr in eligible.items()}
        eligible = _filter_clipped_phones(eligible)

        if kind == "transient":
            frames = [{pid: arr[:n_samples] for pid, arr in eligible.items()}]
        else:
            frame_len = min(n_samples, int(round(self.sustained_frame_s * fs)))
            step = max(1, frame_len // 2)
            hann = np.hanning(frame_len).astype(np.float32)
            frames = []
            pos = 0
            while pos + frame_len <= n_samples and len(frames) < self.sustained_max_frames:
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

        raw_pids = list(event.frames[0].keys())
        if event.kind == "transient":
            raw_event = event.frames[0]
        else:
            n_s = min(len(aligned_audio[pid]) for pid in raw_pids)
            raw_event = {pid: aligned_audio[pid][:n_s] for pid in raw_pids}
        kept = _filter_clipped_phones(raw_event)
        if len(kept) < len(raw_event):
            event.frames = [
                {pid: samples for pid, samples in frame.items() if pid in kept}
                for frame in event.frames
            ]

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


__all__ = ["ClockModel", "SyncEngine", "OnsetDetector", "LocalizationPipeline", "is_window_clipped"]
