"""Session state, geometry, and the optional DSP and classifier integrations."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from uuid import uuid4

import numpy as np

from backend.dsp import ranging
from backend.server.audio_buffer import BUFFER_DURATION_S, PhoneRingBuffer
from backend.server.judge_eval import JudgeEval
from backend.server.recording import Recording
from backend.server.stubs import (
    StubLocalizationPipeline,
    StubOnsetDetector,
    StubSoundClassifier,
    StubSyncEngine,
)
from shared.protocol import (
    FLAG_CLOCK_JUMP,
    CalibrationMode,
    CalibrationResult,
    ClassificationResult,
    DecodedAudioFrame,
    LocalizationResult,
    PhoneAudioSlice,
    PhoneConfig,
    PhoneSyncReport,
    RoomConfig,
    SyncState,
    speed_of_sound,
)

logger = logging.getLogger("uvicorn.error")


def compute_gdop_warning(phones: Iterable[PhoneConfig]) -> str | None:
    coordinates = np.array([(p.x_m, p.y_m) for p in phones if p.connected])
    if len(coordinates) < 3:
        return "Need at least 3 phones for 2D localization (4+ recommended)"
    singular_values = np.linalg.svd(coordinates - coordinates.mean(axis=0), compute_uv=False)
    if singular_values[0] == 0 or singular_values[1] / singular_values[0] < 0.25:
        return "Phones are nearly collinear — spread them around room corners for accurate 2D localization"
    return None


def _discover_class(module: str, name: str, fallback: type) -> type:
    try:
        implementation = getattr(import_module(module), name, None)
    except ModuleNotFoundError as exc:
        if not exc.name or not (exc.name == module or module.startswith(exc.name + ".")):
            raise
        implementation = None
    return implementation or fallback


def _discover(module: str, name: str, fallback: type):
    return _discover_class(module, name, fallback)()


def _discover_classifier():
    """Prefer the winning heuristic backend unless ASCS_CLASSIFIER_BACKEND=hgb is set."""
    cls = _discover_class("backend.classifier.model", "SoundClassifier", StubSoundClassifier)
    if cls is StubSoundClassifier or os.getenv("ASCS_CLASSIFIER_BACKEND") == "hgb":
        return cls()
    return cls(model_path=None)


class CalibrationConflict(Exception):
    """The armed run conflicts with another run or a session change."""


@dataclass
class ArmedCalibration:
    calibration_id: str
    mode: CalibrationMode
    armed_at_s: float
    start_indices: dict[str, int] = field(default_factory=dict)
    result: CalibrationResult | None = None

    def response(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "armed_at_s": self.armed_at_s,
            "expected_duration_s": {
                "lfm_single": 0.5, "lfm_dual": 5.5, "ggwave_hybrid": 5.5,
            }[self.mode],
        }


@dataclass
class ArmedLocate:
    locate_id: str
    order: list[str]                     # phones in emission order (schedule() of these)
    armed_monotonic_s: float
    start_indices: dict[str, int] = field(default_factory=dict)
    apply: bool = True

    def response(self) -> dict:
        slots = ranging.schedule(self.order)
        return {
            "locate_id": self.locate_id,
            "slot_s": ranging.SLOT_S,
            "order": self.order,
            "laptop_slots": [k for k, who in enumerate(slots) if who == ranging.LAPTOP],
            "duration_s": len(slots) * ranging.SLOT_S,
        }


MOVED_WARN_M = 0.10


class Session:
    """Own live state on the asyncio event loop and pass copies to DSP workers."""

    def __init__(self, *, sync_engine=None, localizer=None, classifier=None) -> None:
        self.room = RoomConfig()
        self.phones: dict[str, PhoneConfig] = {}
        self.buffers: dict[str, PhoneRingBuffer] = {}
        self.detections: dict[str, LocalizationResult] = {}
        self.judge = JudgeEval()
        self.recording: Recording | None = None
        self.sync_engine = sync_engine if sync_engine is not None else _discover(
            "backend.dsp.pipeline", "SyncEngine", StubSyncEngine,
        )
        self.localizer = localizer if localizer is not None else _discover(
            "backend.dsp.pipeline", "LocalizationPipeline", StubLocalizationPipeline,
        )
        self.classifier = classifier if classifier is not None else _discover_classifier()
        self.onset_detector_cls = _discover_class("backend.dsp.pipeline", "OnsetDetector", StubOnsetDetector)
        self.onset_detectors: dict[str, object] = {}
        self.onsets: dict[str, list[tuple[int, float]]] = {}  # sample index and estimated onset time
        self.positioned: set[str] = set()  # phones whose position the server owns (CONTRACTS 5)
        self.revision = 0
        self.clock_epoch = 0
        self.calibration_ends: dict[str, int] = {}
        self.calibration: ArmedCalibration | None = None
        self.solving = False
        self.locate: ArmedLocate | None = None
        self.located: dict[str, tuple[float, float]] = {}  # last position found by sound, per phone

    def log_backends(self) -> None:
        logger.info(
            "Transport backends: sync=%s, localization=%s, onsets=%s, classifier=%s (%s)",
            type(self.sync_engine).__name__, type(self.localizer).__name__, self.onset_detector_cls.__name__,
            type(self.classifier).__name__, self.classifier.backend_name,
        )

    def update_room(self, room: RoomConfig) -> None:
        if room != self.room:
            if (room.temperature_c, room.speaker_x_m, room.speaker_y_m) != (
                self.room.temperature_c, self.room.speaker_x_m, self.room.speaker_y_m,
            ):
                for pid in self.phones:
                    self._invalidate_clock(pid, "unsynced")
            self.room = room
            self.revision += 1

    def set_phone(self, phone_id: str, **fields) -> PhoneConfig:
        previous = self.phones.get(phone_id, PhoneConfig(phone_id=phone_id, connected=False))
        updated = previous.model_copy(update=fields)
        self.phones[phone_id] = updated
        if (previous.x_m, previous.y_m) != (updated.x_m, updated.y_m):
            self._invalidate_clock(phone_id, "unsynced")
        return self.phones[phone_id]

    def connect(self, phone_id: str) -> None:
        phone = self.set_phone(phone_id, connected=True, rms=0.0, dropped_frames=0)
        self._replace_buffer(phone_id, phone.sample_rate)
        self.revision += 1

    def disconnect(self, phone_id: str) -> None:
        self.set_phone(phone_id, connected=False, rms=0.0)
        self.buffers.pop(phone_id, None)
        self.onset_detectors.pop(phone_id, None)
        self.onsets.pop(phone_id, None)
        if self.recording is not None:
            self.recording.disconnect(phone_id)
        if self.calibration is not None:
            self.calibration.start_indices.pop(phone_id, None)
        self._invalidate_clock(phone_id, "unsynced")

    def _invalidate_clock(self, phone_id: str, state: SyncState) -> None:
        clock_fn = getattr(self.sync_engine, "clock", None)
        old_ref = getattr(clock_fn(phone_id), "ref_phone_id", None) if callable(clock_fn) else None
        self.sync_engine.mark_clock_jump(phone_id)
        # Losing the reference can re-reference or unsync other phones (CONTRACTS 2).
        rereferenced = old_ref == phone_id
        for pid, phone in self.phones.items():
            if pid == phone_id:
                continue
            if phone.sync_state in ("synced", "drifting") and not self.sync_engine.is_synced(pid):
                self.phones[pid] = phone.model_copy(update={
                    "sync_state": "unsynced", "residual_ms": None, "drift_ppm": None,
                })
            elif rereferenced and callable(clock_fn):
                model = clock_fn(pid)
                if model is not None:
                    new_state: SyncState = "drifting" if model.residual_ms >= 0.5 else "synced"
                    self.phones[pid] = phone.model_copy(update={
                        "sync_state": new_state,
                        "residual_ms": round(model.residual_ms, 4),
                        "drift_ppm": round(model.drift_ppm, 2),
                    })
        if rereferenced:
            self.clock_epoch += 1
        self.calibration_ends.pop(phone_id, None)
        self.phones[phone_id] = self.phones[phone_id].model_copy(update={
            "sync_state": state, "residual_ms": None, "drift_ppm": None,
            "chirp_snr_db": None, "last_sync_ts": None,
        })
        self.revision += 1

    def _clock_jump(self, phone_id: str) -> None:
        if self.calibration is not None:
            self.calibration.start_indices.pop(phone_id, None)
        self._invalidate_clock(phone_id, "clock_jump")

    def _replace_buffer(self, phone_id: str, sample_rate: int) -> PhoneRingBuffer:
        buffer = PhoneRingBuffer(phone_id, sample_rate, self._clock_jump)
        self.buffers[phone_id] = buffer
        self.onset_detectors[phone_id] = self.onset_detector_cls()
        self.onsets[phone_id] = []
        return buffer

    def ingest(self, phone_id: str, frame: DecodedAudioFrame) -> bool:
        received_s = time.monotonic()
        buffer = self.buffers[phone_id]
        rate_changed = frame.sample_rate != buffer.sample_rate
        if rate_changed:
            had_audio = buffer.latest_sample_index > 0
            buffer = self._replace_buffer(phone_id, frame.sample_rate)
            if had_audio and not frame.flags & FLAG_CLOCK_JUMP:
                self._clock_jump(phone_id)
            rate_changed = had_audio
        if not buffer.append(frame):
            return False
        phone = self.set_phone(
            phone_id, sample_rate=frame.sample_rate, rms=buffer.rms, dropped_frames=buffer.dropped_frames,
        )
        if self.recording is not None:
            self.recording.append(phone, frame)
        detector = self.onset_detectors.get(phone_id)
        if detector is not None:
            end_index = frame.sample_index + frame.num_samples
            # shortcut: anchor the end of the PCM batch to its server receive time,
            # use an independent clock or a video capture to measure transport and display latency.
            self.onsets[phone_id].extend(
                (n, received_s - (end_index - n) / frame.sample_rate)
                for n in detector.push(frame.samples_f32, frame.sample_index, frame.sample_rate)
            )
        return rate_changed or bool(frame.flags & FLAG_CLOCK_JUMP)

    def hello_position(self, phone_id: str, fields: dict) -> dict:
        """Apply a phone_hello position only when the server has none (CONTRACTS 5)."""
        if phone_id in self.positioned:
            return {k: v for k, v in fields.items() if k not in ("x_m", "y_m")}
        self.positioned.add(phone_id)
        return fields

    def phone_hull(self) -> list[tuple[float, float]]:
        return [(p.x_m, p.y_m) for p in self.phones.values() if p.connected]

    def active_phones(self) -> dict[str, PhoneConfig]:
        return {pid: phone for pid, phone in self.phones.items() if phone.connected}

    def synced_phones(self) -> dict[str, PhoneConfig]:
        return {
            pid: phone for pid, phone in self.active_phones().items()
            if phone.sync_state in ("synced", "drifting") and self.sync_engine.is_synced(pid)
        }

    def snapshot(self) -> dict:
        active_calibration = self.calibration is not None and self.calibration.result is None
        if self.solving:
            status = "calibrating"
        elif active_calibration:
            status = "armed"
        else:
            status = "synced" if len(self.synced_phones()) >= 2 else "unsynced"
        return {
            "type": "session_state",
            "room": self.room.model_dump(mode="json"),
            "phones": [p.model_dump(mode="json") for p in self.phones.values()],
            "gdop_warning": compute_gdop_warning(self.phones.values()),
            "calibration_status": status,
        }

    def telemetry(self) -> dict:
        return {
            "type": "telemetry",
            "phones": [
                p.model_dump(include={"phone_id", "rms", "connected", "sync_state", "dropped_frames"})
                for p in self.phones.values()
            ],
        }

    def phone_status(self, phone_id: str) -> dict:
        phone = self.phones[phone_id]
        return {
            "type": "phone_status",
            "phone_id": phone_id,
            "synced": phone.sync_state in ("synced", "drifting"),
            "residual_ms": phone.residual_ms,
            "drift_ppm": phone.drift_ppm,
            "state": phone.sync_state,
        }

    def arm(self, mode: CalibrationMode) -> ArmedCalibration:
        if self.solving:
            raise CalibrationConflict("Calibration solve is already running")
        self.calibration = ArmedCalibration(
            uuid4().hex, mode, time.time(),
            {pid: buffer.latest_sample_index for pid, buffer in self.buffers.items()},
        )
        self.revision += 1
        return self.calibration

    async def solve(
        self, calibration_id: str, mode: CalibrationMode, window_s: float,
    ) -> CalibrationResult:
        run = self.calibration
        if run is None or run.calibration_id != calibration_id:
            raise LookupError("Unknown calibration ID")
        if mode != run.mode:
            raise CalibrationConflict("Calibration mode differs from the armed mode")
        if run.result is not None:
            return run.result
        if self.solving:
            raise CalibrationConflict("Calibration solve is already running")

        revision, phones, room = self.revision, self.active_phones(), self.room
        slices: dict[str, PhoneAudioSlice] = {}
        for pid in phones:
            audio = self.buffers[pid].get_recent_slice(min(window_s, BUFFER_DURATION_S))
            crop = min(len(audio.samples_f32), max(0, run.start_indices.get(pid, 0) - audio.start_sample_index))
            slices[pid] = PhoneAudioSlice(
                pid, audio.samples_f32[crop:], audio.start_sample_index + crop, audio.sample_rate,
                audio.valid_mask[crop:] if audio.valid_mask is not None else None,
            )
        # Calibrate a private clock snapshot so a jump or disconnect can invalidate
        # the run without blocking ingestion or letting the worker restore old clocks.
        engine = deepcopy(self.sync_engine)
        if mode in ("lfm_dual", "ggwave_hybrid") and callable(getattr(engine, "clear_history", None)):
            engine.clear_history()
        self.solving = True
        try:
            solve = getattr(engine, "add_chirp_observation", None) or engine.calibrate_from_buffers
            result = await asyncio.to_thread(solve, slices, phones, room, mode)
            if revision != self.revision:
                raise CalibrationConflict("Session changed during calibration. Arm a new run and try again.")

            self.sync_engine = engine
            self.clock_epoch += 1
            self.calibration_ends = {
                pid: audio.start_sample_index + len(audio.samples_f32) for pid, audio in slices.items()
            }
            for pid in phones:
                report = result.per_phone.get(pid, PhoneSyncReport(synced=False))
                state: SyncState = "unsynced"
                if report.synced:
                    state = "drifting" if (report.residual_ms or 0.0) >= 0.5 else "synced"
                self.set_phone(
                    pid, sync_state=state, residual_ms=report.residual_ms,
                    drift_ppm=report.drift_ppm, chirp_snr_db=report.chirp_snr_db,
                    last_sync_ts=time.time() if report.synced else None,
                )
            run.result = result
            self.revision += 1
            recording = self.recording
            if recording is not None and recording.stopped_at_s is None:
                recording.calibrations.append({
                    "offset_s": time.monotonic() - recording.started_monotonic_s,
                    "calibration_id": run.calibration_id,
                    **result.model_dump(mode="json"),
                })
            return result
        finally:
            self.solving = False

    # ---- self-localization ------------------------------------------------------------------
    def arm_locate(self, *, apply: bool = True) -> ArmedLocate:
        if self.solving:
            raise CalibrationConflict("Another solve is already running")
        order = sorted(pid for pid in self.active_phones() if pid in self.buffers)
        if len(order) < ranging.MIN_PHONES:
            raise CalibrationConflict(f"Need at least {ranging.MIN_PHONES} connected phones to locate them")
        self.locate = ArmedLocate(
            uuid4().hex, order, time.monotonic(),
            {pid: self.buffers[pid].latest_sample_index for pid in order}, apply,
        )
        return self.locate

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        return (min(max(x, 0.0), self.room.width_m), min(max(y, 0.0), self.room.height_m))

    async def solve_locate(self, locate_id: str) -> dict:
        run = self.locate
        if run is None or run.locate_id != locate_id:
            raise LookupError("Unknown locate ID")
        if self.solving:
            raise CalibrationConflict("Another solve is already running")
        n_slots = len(ranging.schedule(run.order))
        window_s = min(BUFFER_DURATION_S, n_slots * ranging.SLOT_S + 3.0)
        slices = {}
        for pid in run.order:
            if pid not in self.buffers:
                continue
            audio = self.buffers[pid].get_recent_slice(window_s)
            crop = min(len(audio.samples_f32), max(0, run.start_indices.get(pid, 0) - audio.start_sample_index))
            slices[pid] = (audio.samples_f32[crop:], audio.sample_rate, audio.start_sample_index + crop,
                           audio.valid_mask[crop:] if audio.valid_mask is not None else None)
        prior = {pid: (p.x_m, p.y_m) for pid, p in self.phones.items() if pid in slices}
        speaker = (self.room.speaker_x_m, self.room.speaker_y_m)
        c = speed_of_sound(self.room.temperature_c)

        def work() -> ranging.LayoutResult:
            heard = {pid: ranging.heard_from_audio(s, fs, start, n_slots, valid)
                     for pid, (s, fs, start, valid) in slices.items()}
            return ranging.solve_layout(heard, run.order, speaker, c, prior=prior)

        self.solving = True
        try:
            result = await asyncio.to_thread(work)
        finally:
            self.solving = False
        self.locate = None

        phones_out: dict[str, dict] = {}
        for pid in run.order:
            fix = result.phones.get(pid)
            if fix is None:
                phones_out[pid] = {"status": "failed", "x_m": None, "y_m": None, "rms_m": None, "moved_m": None}
                continue
            x, y = self._clamp(fix.x_m, fix.y_m)
            before = self.located.get(pid)
            moved = float(np.hypot(x - before[0], y - before[1])) if before else None
            phones_out[pid] = {
                "status": fix.status, "x_m": round(x, 3), "y_m": round(y, 3), "rms_m": round(fix.rms_m, 4),
                "heard": fix.heard, "heard_by": fix.heard_by,
                "moved_m": round(moved, 3) if moved is not None else None,
                "moved": moved is not None and moved > MOVED_WARN_M,
            }
            if run.apply and result.success and fix.status != "failed":
                self.set_phone(pid, x_m=round(x, 3), y_m=round(y, 3))
                self.positioned.add(pid)
                self.located[pid] = (x, y)
        if run.apply and result.success:
            self.revision += 1
        return {
            "type": "locate_result", "success": result.success, "applied": run.apply and result.success,
            "anchored": result.anchored, "rms_m": None if result.rms_m is None else round(result.rms_m, 4),
            "error": result.error, "phones": phones_out,
        }

    def turn_layout(self, degrees: float, mirror: bool) -> None:
        """Turn (and optionally mirror) the connected phones about the speaker; the sound fits either way."""
        sx, sy = self.room.speaker_x_m, self.room.speaker_y_m
        a = np.deg2rad(degrees)
        for pid, phone in self.active_phones().items():
            dx, dy = phone.x_m - sx, phone.y_m - sy
            if mirror:
                dx = -dx
            x, y = self._clamp(sx + dx * np.cos(a) - dy * np.sin(a), sy + dx * np.sin(a) + dy * np.cos(a))
            self.set_phone(pid, x_m=round(float(x), 3), y_m=round(float(y), 3))
            if pid in self.located:
                self.located[pid] = (float(x), float(y))
        self.revision += 1

    def record_detection(self, result: LocalizationResult) -> None:
        self.detections.pop(result.event_id, None)
        self.detections[result.event_id] = result
        if len(self.detections) > 100:
            del self.detections[next(iter(self.detections))]
