"""Locked shared protocol, binary PCM frame codec, and data contracts for ASCS Sound Swarm.

Owned by Agent 1 (only modify via team agreement at a checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Physical & Chirp Constants (MUST match frontend/src/shared/chirpSynth.ts)
# ---------------------------------------------------------------------------
CHIRP_F0_HZ: float = 1500.0
CHIRP_F1_HZ: float = 8000.0
CHIRP_DURATION_S: float = 0.15
CHIRP_TUKEY_ALPHA: float = 0.10
CHIRP_AMPLITUDE: float = 0.85
DUAL_CHIRP_GAP_S: float = 5.0

# v2 timing (CONTRACTS 2-4)
AUTO_CHIRP_PERIOD_S: float = 30.0      # periodic single chirp for a long drift baseline
CLOCK_FIT_MAX_AGE_S: float = 600.0     # chirps older than this drop out of the clock fit
PLAYOUT_SLACK_S: float = 0.4           # wait this long after an onset before extracting
EVENT_WINDOW_PRE_S: float = 0.015      # clap window starts 15 ms before global onset
EVENT_WINDOW_POST_S: float = 0.065     # clap window ends 65 ms after global onset
SUSTAINED_WINDOW_S: float = 0.75       # drone window (3 x 250 ms, 50% hop)
CHIRP_SUPPRESS_S: float = 0.3          # no detections within this of a scheduled chirp

# ---------------------------------------------------------------------------
# Binary WebSocket Frame Header (32 bytes, little-endian, 8-byte aligned)
#   0..3   4s  magic ("ASCS")
#   4      B   version (1)
#   5      B   flags (0x01=CLOCK_JUMP, 0x02=CLIPPING)
#   6..7   H   num_samples
#   8..11  I   seq
#   12..15 I   sample_rate (Hz)
#   16..23 Q   sample_index (monotonic AudioWorklet sample count)
#   24..31 d   client_time_ms (telemetry only; NEVER used for TDOA)
#   32..   2*N int16 PCM samples (little-endian)
# ---------------------------------------------------------------------------
FRAME_MAGIC: bytes = b"ASCS"
FRAME_VERSION: int = 1
FRAME_HEADER_FORMAT: str = "<4sBBHIIQd"
FRAME_HEADER_SIZE: int = struct.calcsize(FRAME_HEADER_FORMAT)
assert FRAME_HEADER_SIZE == 32, f"Expected 32-byte header, got {FRAME_HEADER_SIZE}"

FLAG_CLOCK_JUMP: int = 0x01
FLAG_CLIPPING: int = 0x02

SoundClass = Literal["drone", "clap", "other"]
SyncState = Literal["unsynced", "synced", "drifting", "clock_jump"]
# shortcut: "ggwave_hybrid" kept only because legacy DSP paths still accept it; drop with them.
CalibrationMode = Literal["lfm_dual", "lfm_single", "ggwave_hybrid"]
EventKind = Literal["transient", "sustained"]


def speed_of_sound(temperature_c: float = 20.0) -> float:
    """Speed of sound in dry indoor air (m/s)."""
    return 331.3 + 0.606 * temperature_c


@dataclass(slots=True)
class DecodedAudioFrame:
    flags: int
    num_samples: int
    seq: int
    sample_rate: int
    sample_index: int
    client_time_ms: float
    samples_f32: np.ndarray  # shape (num_samples,), dtype float32 in [-1.0, 1.0]


@dataclass(slots=True)
class PhoneAudioSlice:
    phone_id: str
    samples_f32: np.ndarray  # contiguous 1D float32
    start_sample_index: int  # AudioWorklet sample_index of samples_f32[0]
    sample_rate: int
    valid_mask: np.ndarray | None = None  # bool per sample; False denotes a zero-filled gap


def encode_audio_frame(
    samples_f32: np.ndarray,
    seq: int,
    sample_rate: int,
    sample_index: int,
    client_time_ms: float = 0.0,
    flags: int = 0,
) -> bytes:
    """Encode float32 audio [-1, 1] into a 32-byte header + int16 binary WS frame."""
    clipped = np.clip(samples_f32, -1.0, 1.0)
    if np.any(np.abs(samples_f32) >= 0.99):
        flags |= FLAG_CLIPPING
    pcm_i16 = np.rint(clipped * 32767.0).astype("<i2")
    num_samples = int(pcm_i16.shape[0])
    header = struct.pack(
        FRAME_HEADER_FORMAT,
        FRAME_MAGIC,
        FRAME_VERSION,
        flags & 0xFF,
        num_samples,
        seq & 0xFFFFFFFF,
        int(sample_rate) & 0xFFFFFFFF,
        int(sample_index) & 0xFFFFFFFFFFFFFFFF,
        float(client_time_ms),
    )
    return header + pcm_i16.tobytes()


def decode_audio_frame(payload: bytes) -> DecodedAudioFrame:
    """Validate and decode a 32-byte binary PCM frame."""
    if len(payload) < FRAME_HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(payload)} bytes")
    magic, version, flags, num_samples, seq, sample_rate, sample_index, client_time_ms = struct.unpack_from(
        FRAME_HEADER_FORMAT, payload, 0
    )
    if magic != FRAME_MAGIC:
        raise ValueError(f"Invalid frame magic: {magic!r}")
    if version != FRAME_VERSION:
        raise ValueError(f"Unsupported frame version: {version}")
    if num_samples == 0:
        raise ValueError("Empty PCM payload (num_samples == 0)")
    if not (8000 <= sample_rate <= 192000):
        raise ValueError(f"Invalid sample_rate: {sample_rate}")
    expected_len = FRAME_HEADER_SIZE + 2 * num_samples
    if len(payload) != expected_len:
        raise ValueError(f"Payload length mismatch: got {len(payload)}, expected {expected_len}")

    pcm_i16 = np.frombuffer(payload, dtype="<i2", count=num_samples, offset=FRAME_HEADER_SIZE)
    samples_f32 = pcm_i16.astype(np.float32) / 32768.0
    return DecodedAudioFrame(
        flags=flags,
        num_samples=num_samples,
        seq=seq,
        sample_rate=sample_rate,
        sample_index=sample_index,
        client_time_ms=client_time_ms,
        samples_f32=samples_f32,
    )


# ---------------------------------------------------------------------------
# JSON / REST / WebSocket Schemas
# ---------------------------------------------------------------------------
class RoomConfig(BaseModel):
    width_m: float = Field(default=8.0, gt=0.5, le=100.0)
    height_m: float = Field(default=6.0, gt=0.5, le=100.0)
    temperature_c: float = Field(default=20.0, ge=0.0, le=45.0)
    speaker_x_m: float = Field(default=4.0, ge=0.0)
    speaker_y_m: float = Field(default=0.5, ge=0.0)


class PhoneConfig(BaseModel):
    phone_id: str
    label: str = ""
    x_m: float = 0.0
    y_m: float = 0.0
    sample_rate: int = 48000
    connected: bool = True
    rms: float = 0.0
    sync_state: SyncState = "unsynced"
    residual_ms: float | None = None
    drift_ppm: float | None = None
    chirp_snr_db: float | None = None
    last_sync_ts: float | None = None
    dropped_frames: int = 0
    device_info: str = ""


class PhoneSyncReport(BaseModel):
    synced: bool
    residual_ms: float | None = None
    drift_ppm: float | None = None
    chirp_snr_db: float | None = None
    error: str | None = None


class CalibrationResult(BaseModel):
    mode: CalibrationMode
    success: bool
    synced_count: int
    median_residual_ms: float | None = None
    max_residual_ms: float | None = None
    per_phone: dict[str, PhoneSyncReport] = Field(default_factory=dict)


class PairTdoa(BaseModel):
    i: str
    j: str
    tdoa_ms: float
    weight: float
    inlier: bool = True


class Heatmap2D(BaseModel):
    nx: int
    ny: int
    values: list[float]  # row-major ny*nx normalized to [0, 1]


class ClassificationResult(BaseModel):
    sound_class: SoundClass
    is_drone: bool
    confidence: float = Field(ge=0.0, le=1.0)
    probs: dict[str, float]
    backend_name: str = "heuristic"


class LocalizationResult(BaseModel):
    event_id: str
    timestamp_s: float
    x_m: float
    y_m: float
    uncertainty_m: float
    residual_ms: float
    confidence: float = Field(ge=0.0, le=1.0)
    phones_used: list[str]
    pair_tdoas_ms: list[PairTdoa] = Field(default_factory=list)
    heatmap: Heatmap2D | None = None
    sound_class: SoundClass = "other"
    is_drone: bool = False
    class_confidence: float = 0.0
    class_probs: dict[str, float] = Field(default_factory=lambda: {"drone": 0.0, "clap": 0.0, "other": 1.0})
    update_seq: int = 0  # v2: 0 for the first emission, increments on sustained re-emits
    kind: EventKind = "transient"


class EvalTrial(BaseModel):
    trial_idx: int
    label: str
    expected_class: Literal["clap", "drone", "other"] = "clap"
    gt_x_m: float
    gt_y_m: float
    est_x_m: float | None = None
    est_y_m: float | None = None
    error_m: float | None = None
    missed: bool = False
    pred_class: SoundClass | None = None
    class_confidence: float | None = None
    residual_ms: float | None = None
    timestamp_s: float = 0.0


class EvalSummary(BaseModel):
    total: int
    detected: int
    missed: int
    mean_error_m: float | None = None
    median_error_m: float | None = None
    p90_error_m: float | None = None
    max_error_m: float | None = None
