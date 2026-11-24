"""Periodic, sample-clock-aligned audio processing for the live dashboard."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from copy import deepcopy

import numpy as np

from backend.server.session import Session

logger = logging.getLogger(__name__)

TARGET_FS = 48000
WINDOW_S = 1.0
PLAYOUT_SLACK_S = 0.120
PROCESS_INTERVAL_S = 0.080
TELEMETRY_INTERVAL_S = 0.100


class Processor:
    def __init__(
        self,
        session: Session,
        broadcast: Callable[[dict], None],
        *,
        min_phones: int = 3,
        cooldown_s: float = 0.250,
    ) -> None:
        self.session = session
        self.broadcast = broadcast
        self.min_phones = min_phones
        self.cooldown_s = cooldown_s
        self._clock_epoch = -1
        self._last_window_end = -float("inf")
        self._transient_cutoff = -float("inf")
        self._last_drone_end = -float("inf")

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            try:
                await self.process_once()
            except Exception:
                logger.exception("Audio processing failed")
            await asyncio.sleep(max(0.0, PROCESS_INTERVAL_S - (loop.time() - started)))

    async def run_telemetry(self) -> None:
        while True:
            await asyncio.sleep(TELEMETRY_INTERVAL_S)
            self.broadcast(self.session.telemetry())

    async def process_once(self) -> None:
        session = self.session
        revision = session.revision
        if session.clock_epoch != self._clock_epoch:
            self._clock_epoch = session.clock_epoch
            self._last_window_end = self._transient_cutoff = self._last_drone_end = -float("inf")
        phones = session.synced_phones()
        if len(phones) < self.min_phones or (session.calibration is not None and session.calibration.result is None):
            return

        engine = deepcopy(session.sync_engine)
        buffers = {
            pid: session.buffers[pid].get_recent_slice(WINDOW_S + 2 * PLAYOUT_SLACK_S)
            for pid in phones
        }
        calibration_ends = session.calibration_ends.copy()
        room = session.room
        last_end, transient_cutoff = self._last_window_end, self._transient_cutoff

        def analyze():
            ranges = {}
            for pid, audio in buffers.items():
                first = max(audio.start_sample_index, calibration_ends.get(pid, 0))
                last = audio.start_sample_index + len(audio.samples_f32) - 1
                if audio.valid_mask is not None and not audio.valid_mask.all():
                    last_gap = int(np.flatnonzero(~audio.valid_mask)[-1])
                    first = max(first, audio.start_sample_index + last_gap + 1)
                if first < last:
                    ranges[pid] = (
                        engine.sample_to_global_time(pid, first),
                        engine.sample_to_global_time(pid, last),
                    )

            # Take the newest window that a quorum has captured. A lagging phone
            # can participate within the slack or rejoin when its audio catches up.
            for latest in sorted({last for _, last in ranges.values()}, reverse=True):
                end = latest - PLAYOUT_SLACK_S
                if end <= last_end:
                    return last_end, None, None
                eligible = {
                    pid: first for pid, (first, last) in ranges.items()
                    if first <= end - PROCESS_INTERVAL_S and last >= end
                }
                if len(eligible) < self.min_phones:
                    continue
                start = max(max(eligible.values()), end - WINDOW_S, transient_cutoff)
                if end - start < PROCESS_INTERVAL_S:
                    return last_end, None, None
                break
            else:
                return last_end, None, None

            aligned = engine.extract_aligned_window(
                {pid: buffers[pid] for pid in eligible}, start, end, target_fs=TARGET_FS,
            )
            if len(aligned) < self.min_phones:
                return end, None, None
            timestamp = time.time() - PLAYOUT_SLACK_S
            result = session.localizer.detect_and_localize(
                aligned, phones, room, fs=TARGET_FS, timestamp_s=timestamp,
            )
            if result is None:
                return end, None, None
            classification = session.classifier.classify_swarm(aligned, fs=TARGET_FS)
            event_time = end
            if not classification.is_drone:
                # shortcut: use the aligned energy peak as the transient marker until
                # the localization facade exposes its onset sample.
                size = min(map(len, aligned.values()))
                power = np.sum([np.square(samples[:size]) for samples in aligned.values()], axis=0)
                event_time = start + int(np.argmax(power)) / TARGET_FS
                timestamp -= end - event_time
            return end, event_time, result.model_copy(update={
                "timestamp_s": timestamp,
                "sound_class": classification.sound_class,
                "is_drone": classification.is_drone,
                "class_confidence": classification.confidence,
                "class_probs": classification.probs,
            })

        end, event_time, result = await asyncio.to_thread(analyze)
        if revision != session.revision:
            return
        self._last_window_end = end
        if result is None or event_time is None:
            return
        if result.is_drone:
            if end - self._last_drone_end < self.cooldown_s:
                return
            self._last_drone_end = end
        else:
            self._transient_cutoff = event_time + self.cooldown_s

        session.record_detection(result)
        self.broadcast({"type": "detection", **result.model_dump(mode="json")})
