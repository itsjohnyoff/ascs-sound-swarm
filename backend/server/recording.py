"""Disk-backed phone audio capture in the DSP replay archive format."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import numpy as np

from shared.protocol import FLAG_CLOCK_JUMP, DecodedAudioFrame, PhoneConfig, RoomConfig

logger = logging.getLogger(__name__)
MAX_GAP_S = 120  # Start a new epoch for implausible sample-counter jumps.


class _Track:
    def __init__(self, root: Path, index: int, phone: PhoneConfig, frame: DecodedAudioFrame) -> None:
        self.phone = phone
        self.sample_rate = frame.sample_rate
        self.start_index = self.end_index = frame.sample_index
        self.captured_samples = 0
        self.pcm_path = root / f"{index}.f32"
        self.valid_path = root / f"{index}.bool"
        self.pcm = self.pcm_path.open("wb")
        try:
            self.valid = self.valid_path.open("wb")
        except OSError:
            self.pcm.close()
            raise

    def append(self, phone: PhoneConfig, frame: DecodedAudioFrame) -> None:
        # Seeking past EOF leaves a zero-filled hole in both files.
        if frame.sample_index != self.end_index:
            offset = frame.sample_index - self.start_index
            self.pcm.seek(offset * 4)
            self.valid.seek(offset)
        self.pcm.write(frame.samples_f32.astype("<f4", copy=False).tobytes())
        self.valid.write(b"\x01" * frame.num_samples)
        self.end_index = frame.sample_index + frame.num_samples
        self.captured_samples += frame.num_samples
        self.phone = phone

    def close(self) -> None:
        try:
            self.pcm.close()
        finally:
            self.valid.close()

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.memmap(self.pcm_path, dtype="<f4", mode="r"), np.memmap(self.valid_path, dtype=bool, mode="r")


class Recording:
    def __init__(self, path: Path, room: RoomConfig) -> None:
        self.path = path
        self.room = room
        self.started_at_s = time.time()
        self.stopped_at_s: float | None = None
        self.saving = False
        self.failed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._staging = TemporaryDirectory(prefix=".capture-", dir=path.parent)
        self._root = Path(self._staging.name)
        self._tracks: dict[str, list[_Track]] = {}
        self._active: dict[str, _Track] = {}
        self._next_track = 0

    @property
    def has_audio(self) -> bool:
        return bool(self._tracks)

    def append(self, phone: PhoneConfig, frame: DecodedAudioFrame) -> None:
        if self.stopped_at_s is not None or self.failed:
            return
        pid = phone.phone_id
        try:
            track = self._active.get(pid)
            if track is not None and (
                frame.flags & FLAG_CLOCK_JUMP
                or track.sample_rate != frame.sample_rate
                or not 0 <= frame.sample_index - track.end_index <= MAX_GAP_S * frame.sample_rate
            ):
                self.disconnect(pid)
                track = None
            if track is None:
                track = _Track(self._root, self._next_track, phone, frame)
                self._next_track += 1
                self._tracks.setdefault(pid, []).append(track)
                self._active[pid] = track
            track.append(phone, frame)
        except OSError:
            self.failed = True
            logger.exception("Could not spool audio for %s", self.path)

    def disconnect(self, phone_id: str) -> None:
        if self.stopped_at_s is not None:
            return
        track = self._active.pop(phone_id, None)
        if track is not None:
            try:
                track.close()
            except OSError:
                self.failed = True
                logger.exception("Could not close audio track for %s", self.path)

    def _close_tracks(self) -> None:
        error = None
        for track in self._active.values():
            try:
                track.close()
            except OSError as exc:
                error = exc
        self._active.clear()
        if error is not None:
            self.failed = True
            raise error

    def save(self) -> None:
        self._close_tracks()
        self._write_archive()

    def _write_archive(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        phones: dict[str, dict] = {}
        segments: dict[str, list[dict]] = {}
        try:
            for original_id, tracks in self._tracks.items():
                pid = quote(original_id, safe="")
                primary = max(tracks, key=lambda track: track.captured_samples / track.sample_rate)
                phones[pid] = primary.phone.model_dump(
                    mode="json", include={"phone_id", "label", "x_m", "y_m"},
                )
                arrays[f"{pid}_start_index"] = np.asarray(primary.start_index, dtype=np.uint64)
                arrays[f"{pid}_fs"] = np.asarray(primary.sample_rate, dtype=np.int32)
                segments[pid] = []
                for i, track in enumerate(tracks):
                    if track is primary:
                        pcm_key, mask_key = f"{pid}_pcm", f"{pid}_valid"
                    else:
                        # Replay sees one clock epoch per phone. Keep the other audio
                        # accessible in the archive without treating it as extra phones.
                        pcm_key, mask_key = f"{pid}_epoch_{i}_samples", f"{pid}_epoch_{i}_mask"
                    arrays[pcm_key], arrays[mask_key] = track.arrays()
                    segments[pid].append({
                        "start_index": track.start_index, "sample_rate": track.sample_rate,
                        "samples_key": pcm_key, "valid_key": mask_key,
                        "x_m": track.phone.x_m, "y_m": track.phone.y_m,
                    })

            arrays["meta"] = np.asarray(json.dumps({
                "room": self.room.model_dump(mode="json"), "phones": phones, "segments": segments,
                "started_at_s": self.started_at_s, "stopped_at_s": self.stopped_at_s,
            }))
            temporary_archive = self._root / "recording.npz"
            with temporary_archive.open("wb") as output:
                np.savez(output, **arrays)
            # Publish atomically without clobbering an existing capture of the same name.
            os.link(temporary_archive, self.path)
        finally:
            for arr in arrays.values():
                if isinstance(arr, np.memmap) and arr._mmap is not None:
                    arr._mmap.close()

    def close(self) -> None:
        try:
            self._close_tracks()
        finally:
            self._staging.cleanup()
