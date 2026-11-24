"""PCM ingestion and indexed per-phone audio snapshots."""

from collections.abc import Callable
from threading import Lock

import numpy as np

from shared.protocol import (
    FLAG_CLIPPING,
    FLAG_CLOCK_JUMP,
    DecodedAudioFrame,
    PhoneAudioSlice,
    decode_audio_frame as decode_audio_frame,
)


BUFFER_DURATION_S = 12
SEQUENCE_MASK = (1 << 32) - 1
SEQUENCE_HALF_RANGE = 1 << 31


class PhoneRingBuffer:
    """Keep the latest audio and sample validity in fixed-capacity circular arrays."""

    def __init__(
        self,
        phone_id: str,
        sample_rate: int,
        on_clock_jump: Callable[[str], None] | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.phone_id = phone_id
        self.sample_rate = sample_rate
        self.latest_sample_index = 0
        self.dropped_frames = 0
        self.clipping_count = 0
        self.rms = 0.0

        self._capacity = sample_rate * BUFFER_DURATION_S
        self._samples = np.empty(self._capacity, dtype=np.float32)
        self._valid = np.empty(self._capacity, dtype=bool)
        self._available = 0
        self._last_seq: int | None = None
        self._on_clock_jump = on_clock_jump
        self._lock = Lock()

    def append(self, frame: DecodedAudioFrame) -> bool:
        """Append a decoded frame and return whether it belongs to the stream."""
        if frame.sample_rate != self.sample_rate:
            return False

        clock_jump = bool(frame.flags & FLAG_CLOCK_JUMP)
        with self._lock:
            gap_size = 0
            if clock_jump:
                self._available = 0
                self.dropped_frames = 0
            elif self._last_seq is not None:
                distance = (frame.seq - self._last_seq) & SEQUENCE_MASK
                if not 0 < distance < SEQUENCE_HALF_RANGE:
                    return False
                if frame.sample_index < self.latest_sample_index:
                    return False

                gap_size = min(frame.sample_index - self.latest_sample_index, self._capacity)
                self.dropped_frames += max(distance - 1, int(gap_size > 0))
                if gap_size:
                    gap_start = (frame.sample_index - gap_size) % self._capacity
                    first_gap = min(gap_size, self._capacity - gap_start)
                    self._samples[gap_start:gap_start + first_gap] = 0
                    self._samples[:gap_size - first_gap] = 0
                    self._valid[gap_start:gap_start + first_gap] = False
                    self._valid[:gap_size - first_gap] = False

            end = frame.sample_index + frame.num_samples
            count = min(frame.num_samples, self._capacity)
            start = (end - count) % self._capacity
            first = min(count, self._capacity - start)
            samples = frame.samples_f32[-count:]
            self._samples[start:start + first] = samples[:first]
            self._samples[:count - first] = samples[first:]

            self._valid[start:start + first] = True
            self._valid[:count - first] = True
            self._available = min(self._available + gap_size + count, self._capacity)
            self.latest_sample_index = end
            self._last_seq = frame.seq
            self.rms = float(np.sqrt(np.mean(np.square(frame.samples_f32))))
            self.clipping_count += int(bool(frame.flags & FLAG_CLIPPING))

        if clock_jump and self._on_clock_jump is not None:
            self._on_clock_jump(self.phone_id)
        return True

    def get_recent_slice(self, duration_s: float) -> PhoneAudioSlice:
        """Copy recent audio and its validity mask in chronological order."""
        with self._lock:
            count = max(0, min(self._available, int(duration_s * self.sample_rate)))
            start_index = self.latest_sample_index - count
            start = start_index % self._capacity
            first = min(count, self._capacity - start)

            samples = np.empty(count, dtype=np.float32)
            samples[:first] = self._samples[start:start + first]
            samples[first:] = self._samples[:count - first]
            valid = np.empty(count, dtype=bool)
            valid[:first] = self._valid[start:start + first]
            valid[first:] = self._valid[:count - first]

        return PhoneAudioSlice(self.phone_id, samples, start_index, self.sample_rate, valid)
