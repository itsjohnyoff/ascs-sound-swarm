"""Adaptive transient (clap) vs sustained (drone) acoustic event trigger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class EventWindow:
    kind: Literal["transient", "sustained"]
    frames: list[dict[str, np.ndarray]]
    onset_offset_s: float
    peak_snr_db: float


class AdaptiveEventTrigger:
    """Detects active sound events and crops direct-path or multi-frame windows."""

    def __init__(self) -> None:
        self._noise_floor: dict[str, float] = {}

    def detect(
        self,
        aligned_audio: dict[str, np.ndarray],
        fs: int = 48000,
    ) -> EventWindow | None:
        if len(aligned_audio) < 2:
            return None

        pids = list(aligned_audio.keys())
        n_samples = min(len(aligned_audio[pid]) for pid in pids)
        if n_samples < int(0.05 * fs):
            return None

        hop = max(1, int(round(0.010 * fs)))
        n_frames = max(1, n_samples // hop)
        required_active = min(3, len(pids))

        active_phones = 0
        best_snr_db = 0.0
        crest_factors: list[float] = []
        active_durations_s: list[float] = []
        onset_indices: list[int] = []

        for pid in pids:
            x = aligned_audio[pid][:n_samples].astype(np.float64)
            # 10 ms frame RMS envelope
            trimmed = x[: n_frames * hop].reshape(n_frames, hop)
            frame_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))
            peak_rms = float(np.max(frame_rms))
            win_rms = float(np.sqrt(np.mean(x * x))) + 1e-12
            peak_abs = float(np.max(np.abs(x)))

            # Lazy floor seeded from quietest 20% of first quiet frames; capped below 0.015 so
            # continuous drone windows never inflate the background floor into suppressing events.
            p20 = float(np.percentile(frame_rms, 20))
            prev_floor = self._noise_floor.get(pid)
            if prev_floor is None:
                floor = min(max(p20, 1e-5), 0.015)
            elif p20 < prev_floor * 2.5:
                floor = 0.9 * prev_floor + 0.1 * max(p20, 1e-5)
            else:
                floor = prev_floor
            self._noise_floor[pid] = floor

            snr_db = 20.0 * float(np.log10(max(peak_rms, 1e-12) / max(floor, 1e-6)))
            if snr_db >= 8.0 and peak_rms >= 0.004:
                active_phones += 1
                best_snr_db = max(best_snr_db, snr_db)
                crest_factors.append(peak_abs / win_rms)

                active_mask = frame_rms >= max(0.25 * peak_rms, 2.5 * floor)
                active_durations_s.append(float(np.sum(active_mask)) * hop / float(fs))

                # Earliest frame crossing 30% of peak RMS
                idx_hits = np.flatnonzero(frame_rms >= 0.30 * peak_rms)
                if len(idx_hits) > 0:
                    onset_indices.append(int(idx_hits[0]) * hop)

        if active_phones < required_active:
            return None

        med_crest = float(np.median(crest_factors))
        med_dur = float(np.median(active_durations_s))

        if med_crest >= 5.5 or med_dur <= 0.12:
            # Transient (clap): crop tight 80 ms direct-path window [-15 ms, +65 ms] around earliest onset
            earliest_onset = min(onset_indices) if onset_indices else 0
            start_i = max(0, earliest_onset - int(round(0.015 * fs)))
            end_i = min(n_samples, earliest_onset + int(round(0.065 * fs)))
            if end_i - start_i < int(round(0.040 * fs)):
                start_i = max(0, end_i - int(round(0.080 * fs)))
            cropped = {pid: aligned_audio[pid][start_i:end_i] for pid in pids}
            return EventWindow(
                kind="transient",
                frames=[cropped],
                onset_offset_s=start_i / float(fs),
                peak_snr_db=best_snr_db,
            )

        # Sustained (drone): slice up to 3 overlapping 250 ms Hann-windowed frames (50% hop)
        frame_len = min(n_samples, int(round(0.250 * fs)))
        step = max(1, frame_len // 2)
        hann = np.hanning(frame_len).astype(np.float32)

        frames: list[dict[str, np.ndarray]] = []
        pos = 0
        while pos + frame_len <= n_samples and len(frames) < 3:
            frames.append({pid: aligned_audio[pid][pos : pos + frame_len] * hann for pid in pids})
            pos += step
        if not frames:
            w = np.hanning(n_samples).astype(np.float32)
            frames.append({pid: aligned_audio[pid][:n_samples] * w for pid in pids})

        return EventWindow(
            kind="sustained",
            frames=frames,
            onset_offset_s=0.0,
            peak_snr_db=best_snr_db,
        )
