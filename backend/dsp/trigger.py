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

    def __init__(
        self,
        *,
        sustained_frame_s: float = 0.250,
        sustained_max_frames: int = 3,
        transient_pre_s: float = 0.015,
        transient_post_s: float = 0.065,
    ) -> None:
        self.sustained_frame_s = sustained_frame_s
        self.sustained_max_frames = sustained_max_frames
        self.transient_pre_s = transient_pre_s
        self.transient_post_s = transient_post_s
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
            # Transient (clap): crop tight direct-path window [-pre, +post] around earliest onset
            earliest_onset = min(onset_indices) if onset_indices else 0
            start_i = max(0, earliest_onset - int(round(self.transient_pre_s * fs)))
            end_i = min(n_samples, earliest_onset + int(round(self.transient_post_s * fs)))
            if end_i - start_i < int(round(0.040 * fs)):
                start_i = max(0, end_i - int(round((self.transient_pre_s + self.transient_post_s) * fs)))
            cropped = {pid: aligned_audio[pid][start_i:end_i] for pid in pids}
            return EventWindow(
                kind="transient",
                frames=[cropped],
                onset_offset_s=start_i / float(fs),
                peak_snr_db=best_snr_db,
            )

        # Sustained (drone): slice overlapping Hann-windowed frames (50% hop)
        frame_len = min(n_samples, int(round(self.sustained_frame_s * fs)))
        step = max(1, frame_len // 2)
        hann = np.hanning(frame_len).astype(np.float32)

        frames: list[dict[str, np.ndarray]] = []
        pos = 0
        while pos + frame_len <= n_samples and len(frames) < self.sustained_max_frames:
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


class OnsetDetector:
    """v2 stateful per-phone onset detector for event-driven server ingestion."""

    def __init__(
        self,
        hop_s: float = 0.005,
        snr_db_threshold: float = 10.0,
        refractory_s: float = 0.15,
        min_rms: float = 0.0012,
    ) -> None:
        self.hop_s = hop_s
        self.snr_db_threshold = snr_db_threshold
        self.refractory_s = refractory_s
        self.min_rms = min_rms
        self._floor: float | None = None
        self._last_onset_idx: int | None = None
        self._tail: np.ndarray = np.empty(0, dtype=np.float32)
        self._tail_valid: np.ndarray = np.empty(0, dtype=bool)
        self._next_sample_idx: int | None = None

    def push(
        self,
        samples: np.ndarray,
        start_sample_index: int,
        fs: int,
        valid_mask: np.ndarray | None = None,
    ) -> list[int]:
        """Push an incoming audio chunk and return absolute sample indices of detected onsets."""
        if len(samples) == 0:
            return []
        x = np.asarray(samples, dtype=np.float32)
        v = (
            np.asarray(valid_mask, dtype=bool)
            if valid_mask is not None
            else np.ones(len(x), dtype=bool)
        )
        if len(v) != len(x):
            raise ValueError(f"valid_mask length ({len(v)}) != samples length ({len(x)})")

        if self._next_sample_idx is None or start_sample_index != self._next_sample_idx:
            # Discontinuity or first push: reset leftover tail cleanly
            buf = x
            buf_v = v
            base_idx = int(start_sample_index)
        else:
            buf = np.concatenate([self._tail, x]) if len(self._tail) else x
            buf_v = np.concatenate([self._tail_valid, v]) if len(self._tail_valid) else v
            base_idx = int(start_sample_index) - len(self._tail)

        hop = max(1, int(round(self.hop_s * fs)))
        refractory_n = int(round(self.refractory_s * fs))
        n_hops = len(buf) // hop
        onsets: list[int] = []

        for h in range(n_hops):
            s = h * hop
            e = s + hop
            if not bool(np.all(buf_v[s:e])):
                continue
            seg = buf[s:e].astype(np.float64)
            rms = float(np.sqrt(np.mean(seg * seg)))
            if self._floor is None:
                self._floor = min(max(rms, 1e-5), 0.015)

            snr_db = 20.0 * float(np.log10(max(rms, 1e-12) / max(self._floor, 1e-6)))
            hop_abs_idx = base_idx + s

            if snr_db > self.snr_db_threshold and rms >= self.min_rms:
                if self._last_onset_idx is None or (hop_abs_idx - self._last_onset_idx) >= refractory_n:
                    self._last_onset_idx = hop_abs_idx
                    onsets.append(hop_abs_idx)
            elif snr_db <= self.snr_db_threshold:
                # Never poison the background floor on frames that already exceed the SNR gate;
                # decay quickly on quieter frames and cap any upward creep during weak signal tails.
                if rms <= self._floor:
                    self._floor = 0.95 * self._floor + 0.05 * max(rms, 1e-5)
                else:
                    self._floor = min(0.015, 0.99 * self._floor + 0.01 * min(rms, 2.0 * self._floor))

        rem_start = n_hops * hop
        self._tail = buf[rem_start:].copy()
        self._tail_valid = buf_v[rem_start:].copy()
        self._next_sample_idx = int(start_sample_index) + len(x)
        return onsets
