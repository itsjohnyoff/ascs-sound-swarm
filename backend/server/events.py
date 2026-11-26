"""Event-driven detection (CONTRACTS 4).

Per-phone onsets -> global time -> group (>= 3 phones within room diagonal / c + 5 ms)
-> wait for playout slack -> one window -> one localize + one classify -> one detection.

`EventGrouper`, `decide_kind` and `extract_event` are pure so the live loop, the spike
analysis and the benchmark all run the exact same logic.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from backend.dsp.tracker import MovingTracker2D
from shared.protocol import (
    CHIRP_DURATION_S,
    CHIRP_SUPPRESS_S,
    EVENT_WINDOW_POST_S,
    EVENT_WINDOW_PRE_S,
    PLAYOUT_SLACK_S,
    SUSTAINED_WINDOW_S,
    EventKind,
    LocalizationResult,
    PhoneAudioSlice,
    PhoneConfig,
    RoomConfig,
    speed_of_sound,
)

TARGET_FS = 48000
SUSTAIN_CHECK_S = 0.25        # energy still high this long after onset -> sustained
SUSTAIN_THRESHOLD_DB = 10.0
SUSTAIN_REEMIT_S = 0.25
SUSTAIN_MAX_UPDATES = 240     # track sustained sources up to ~60 s on a single event_id
FRAME = 2048


def max_spread_s(room: RoomConfig) -> float:
    return math.hypot(room.width_m, room.height_m) / speed_of_sound(room.temperature_c) + 0.005


@dataclass
class PendingEvent:
    t0: float
    onsets: dict[str, float] = field(default_factory=dict)
    onset_monotonic_s: float | None = None


class EventGrouper:
    """Group per-phone global onset times into events."""

    def __init__(self, max_spread_s: float, min_phones: int = 3) -> None:
        self.max_spread_s = max_spread_s
        self.min_phones = min_phones
        self.pending: list[PendingEvent] = []
        self._recent: list[PendingEvent] = []  # emitted, absorbs late onsets of the same sound
        self.hold_until = -math.inf  # a sustained sound is still playing until this global time

    def add(self, phone_id: str, t: float, onset_monotonic_s: float | None = None) -> None:
        if t < self.hold_until:
            return
        for ev in self._recent + self.pending:
            if abs(t - ev.t0) <= self.max_spread_s and phone_id not in ev.onsets:
                ev.onsets[phone_id] = t
                if t < ev.t0:
                    ev.t0 = t
                    ev.onset_monotonic_s = onset_monotonic_s
                return
            if abs(t - ev.t0) <= self.max_spread_s and ev in self._recent:
                return
        self.pending.append(PendingEvent(t, {phone_id: t}, onset_monotonic_s))

    def hold(self, until: float) -> None:
        """Ignore onsets (and drop pending clusters) until a sustained sound has ended."""
        self.hold_until = max(self.hold_until, until)
        self.pending = [ev for ev in self.pending if ev.t0 >= self.hold_until]

    def pop_ready(self, buffered_until: Callable[[str], float | None], slack_s: float = PLAYOUT_SLACK_S
                  ) -> list[PendingEvent]:
        ready, keep = [], []
        for ev in self.pending:
            ends = [buffered_until(pid) for pid in ev.onsets]
            passed = all(t is not None and t >= ev.t0 + slack_s for t in ends)
            if passed and len(ev.onsets) >= self.min_phones:
                ready.append(ev)
            elif not passed:
                keep.append(ev)
        self.pending = keep
        if ready:  # keep emitted events for 5 s so their late onsets are absorbed, not re-detected
            newest = max(r.t0 for r in ready)
            self._recent = [ev for ev in self._recent + ready if ev.t0 >= newest - 5.0]
        return sorted(ready, key=lambda e: e.t0)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def decide_kind(
    engine,
    slices: dict[str, PhoneAudioSlice],
    t0: float,
    fs: int = TARGET_FS,
    floor_rms: dict[str, float] | None = None,
) -> EventKind | None:
    """Transient unless energy 250 ms after onset is still > 10 dB above the pre-onset floor."""
    aligned = engine.extract_aligned_window(slices, t0 - 0.12, t0 + SUSTAIN_CHECK_S + 0.05, target_fs=fs)
    if not aligned:
        return None
    ratios = []
    for pid, x in aligned.items():
        noise = _rms(x[: int(0.1 * fs)])
        if floor_rms is not None:
            fb = floor_rms.get(pid)
            if fb is not None and fb > 0.0:
                noise = min(noise, fb)
        late = _rms(x[int((0.12 + SUSTAIN_CHECK_S) * fs):])
        ratios.append(20 * math.log10(max(late, 1e-9) / max(noise, 1e-9)))
    return "sustained" if float(np.median(ratios)) > SUSTAIN_THRESHOLD_DB else "transient"


def event_window(t0: float, kind: EventKind, update_seq: int = 0) -> tuple[float, float]:
    if kind == "transient":
        return t0 - EVENT_WINDOW_PRE_S, t0 + EVENT_WINDOW_POST_S
    start = t0 + update_seq * SUSTAIN_REEMIT_S
    return start, start + SUSTAINED_WINDOW_S


def extract_event(engine, slices: dict[str, PhoneAudioSlice], t0: float, kind: EventKind, update_seq: int = 0,
                  fs: int = TARGET_FS, min_phones: int = 3) -> dict[str, np.ndarray]:
    """Aligned window per phone. Gap phones are dropped; severely clipped phones too while >= min_phones remain."""
    lo, hi = event_window(t0, kind, update_seq)
    aligned = engine.extract_aligned_window(slices, lo, hi, target_fs=fs)  # already drops gap phones
    clean = {
        pid: x for pid, x in aligned.items()
        if len(x) == 0 or float(np.mean(np.abs(x) >= 0.99)) < 0.01
    }
    return clean if len(clean) >= min_phones else aligned


def still_sustained(
    engine,
    slices,
    t0: float,
    update_seq: int,
    fs: int = TARGET_FS,
    pre_rms: dict[str, float] | None = None,
) -> bool:
    _, hi = event_window(t0, "sustained", update_seq)
    cur = engine.extract_aligned_window(slices, hi - 0.25, hi, target_fs=fs)
    if pre_rms:
        common = set(pre_rms) & set(cur)
        if not common:
            return False
        ratios = [20 * math.log10(max(_rms(cur[p]), 1e-9) / max(pre_rms[p], 1e-9)) for p in common]
        return float(np.median(ratios)) > SUSTAIN_THRESHOLD_DB
    pre = engine.extract_aligned_window(slices, t0 - 0.12, t0 - 0.02, target_fs=fs)
    common = set(pre) & set(cur)
    if not common:
        return False
    ratios = [20 * math.log10(max(_rms(cur[p]), 1e-9) / max(_rms(pre[p]), 1e-9)) for p in common]
    return float(np.median(ratios)) > SUSTAIN_THRESHOLD_DB


def sustained_hold_until(t0: float, last_seq: int) -> float:
    # shortcut: a new sound within 0.5 s after a drone's last window is ignored; split
    # by onset energy if judges play claps over drones.
    return event_window(t0, "sustained", last_seq)[1] + max(SUSTAIN_REEMIT_S, 0.5)


# ---------------------------------------------------------------------------
# Offline: same logic over whole recordings (spike analysis, benchmark)
# ---------------------------------------------------------------------------
@dataclass
class OfflineDetection:
    t0: float
    kind: EventKind
    onsets: dict[str, float]
    result: LocalizationResult | None
    aligned: dict[str, np.ndarray]


def chirp_suppression(engine) -> dict[str, list[tuple[float, float]]]:
    """Local sample ranges around every chirp each phone heard."""
    out: dict[str, list[tuple[float, float]]] = {}
    fitter = getattr(engine, "_v2_fitter", None)
    for em in getattr(fitter, "emissions", []):
        for pid, obs in em.obs_by_phone.items():
            pad = CHIRP_SUPPRESS_S * obs.sample_rate
            out.setdefault(pid, []).append(
                (obs.sample_index - pad, obs.sample_index + CHIRP_DURATION_S * obs.sample_rate + pad))
    return out


def find_events_offline(slices: dict[str, PhoneAudioSlice], phones: dict[str, PhoneConfig], room: RoomConfig,
                        *, engine=None, localizer=None, classifier=None, min_phones: int = 3
                        ) -> list[OfflineDetection]:
    from backend.dsp.pipeline import LocalizationPipeline, OnsetDetector, SyncEngine
    from backend.server.session import _discover_classifier

    classifier = classifier or _discover_classifier()

    if engine is None:
        engine = SyncEngine()
        engine.add_chirp_observation(slices, phones, room, "lfm_single")
    localizer = localizer or LocalizationPipeline()
    suppress = chirp_suppression(engine)
    onsets: list[tuple[float, str]] = []
    for pid, s in slices.items():
        if not engine.is_synced(pid):
            continue
        det = OnsetDetector()
        for k in range(0, len(s.samples_f32), FRAME):
            mask = None if s.valid_mask is None else s.valid_mask[k:k + FRAME]
            for n in det.push(s.samples_f32[k:k + FRAME], s.start_sample_index + k, s.sample_rate, mask):
                if any(a <= n <= b for a, b in suppress.get(pid, [])):
                    continue
                t = engine.to_global(pid, n)
                if t is not None:
                    onsets.append((t, pid))
    grouper = EventGrouper(max_spread_s(room), min_phones)
    ends = {pid: engine.to_global(pid, s.start_sample_index + len(s.samples_f32) - 1) for pid, s in slices.items()}
    if all(t is None for t in ends.values()):
        return []  # nothing synced
    data_end = min(t for t in ends.values() if t is not None)
    out: list[OfflineDetection] = []

    def drain(horizon: float) -> None:
        for ev in grouper.pop_ready(lambda pid: horizon, slack_s=PLAYOUT_SLACK_S):
            kind = decide_kind(engine, slices, ev.t0)
            if kind is None:
                continue
            aligned = extract_event(engine, slices, ev.t0, kind, min_phones=min_phones)
            result = None
            if len(aligned) >= min_phones:
                result = localizer.localize(aligned, {p: phones[p] for p in aligned}, room, kind, fs=TARGET_FS)
            if result is not None:
                cls = classifier.classify_swarm(aligned, fs=TARGET_FS)
                result = result.model_copy(update={
                    "kind": kind, "sound_class": cls.sound_class, "is_drone": cls.is_drone,
                    "class_confidence": cls.confidence, "class_probs": cls.probs})
            out.append(OfflineDetection(ev.t0, kind, dict(ev.onsets), result, aligned))
            if kind == "sustained":
                seq = 1
                while (seq < SUSTAIN_MAX_UPDATES and event_window(ev.t0, kind, seq)[1] <= data_end
                       and still_sustained(engine, slices, ev.t0, seq)):
                    seq += 1
                grouper.hold(sustained_hold_until(ev.t0, seq - 1))

    # Replay onsets in time order so holds and slack behave exactly as live.
    for t, pid in sorted(onsets):
        drain(t)
        grouper.add(pid, t)
    drain(math.inf)
    return out


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------
PROCESS_INTERVAL_S = 0.05
TELEMETRY_INTERVAL_S = 0.1
LIVE_SLICE_S = 4.0


@dataclass
class _Sustained:
    event_id: str
    t0: float
    phones_used: set[str]
    seq: int = 0
    pre_rms: dict[str, float] = field(default_factory=dict)
    ema_xy: tuple[float, float] | None = None
    tracker: MovingTracker2D | None = None
    drone_votes: int = 0
    total_votes: int = 0
    drone_conf: float = 0.0
    drone_probs: dict[str, float] | None = None


class EventLoop:
    """Turns ingested onsets into detections on the asyncio loop; DSP runs in a worker thread."""

    def __init__(self, session, broadcast: Callable[[dict], None], *, min_phones: int = 3) -> None:
        self.session = session
        self.broadcast = broadcast
        self.min_phones = min_phones
        self.grouper = EventGrouper(max_spread_s(session.room), min_phones)
        self._epoch = -1
        self._counter = 0
        self._sustained: _Sustained | None = None

    async def run(self) -> None:
        import asyncio
        import logging

        while True:
            try:
                await self.process_once()
            except Exception:
                logging.getLogger(__name__).exception("Event processing failed")
            await asyncio.sleep(PROCESS_INTERVAL_S)

    async def run_telemetry(self) -> None:
        import asyncio

        while True:
            await asyncio.sleep(TELEMETRY_INTERVAL_S)
            self.broadcast(self.session.telemetry())

    def _collect_onsets(self, engine) -> None:
        s = self.session
        armed = s.calibration is not None and s.calibration.result is None
        for pid, pending in s.onsets.items():
            local, s.onsets[pid] = pending, []
            if armed or not engine.is_synced(pid) or pid not in s.buffers:
                continue
            fs = s.buffers[pid].sample_rate
            cutoff = s.calibration_ends.get(pid, -math.inf) + CHIRP_SUPPRESS_S * fs
            for n, estimated_onset_s in local:
                t = engine.to_global(pid, n) if n >= cutoff else None
                if t is not None:
                    self.grouper.add(pid, t, estimated_onset_s)

    async def process_once(self) -> None:
        import asyncio

        s = self.session
        if s.clock_epoch != self._epoch:  # new calibration: old onsets live on an old timeline
            self._epoch = s.clock_epoch
            self.grouper = EventGrouper(max_spread_s(s.room), self.min_phones)
            self._sustained = None
        engine = s.sync_engine
        self._collect_onsets(engine)
        synced = s.synced_phones()
        if len(synced) < self.min_phones:
            return

        def buffered_until(pid: str) -> float | None:
            buf = s.buffers.get(pid)
            return None if buf is None else engine.to_global(pid, buf.latest_sample_index)

        jobs: list[tuple[str, float, EventKind | None, int, set[str], float | None]] = []
        sus = self._sustained
        if sus is not None:
            end = event_window(sus.t0, "sustained", sus.seq + 1)[1]
            if all((buffered_until(p) is not None and buffered_until(p) >= end) for p in sus.phones_used if p in s.buffers):
                jobs.append((sus.event_id, sus.t0, "sustained", sus.seq + 1, sus.phones_used, None))
        for ev in self.grouper.pop_ready(buffered_until):
            self._counter += 1
            jobs.append((f"evt-{self._counter:04d}", ev.t0, None, 0, set(synced), ev.onset_monotonic_s))
        if not jobs:
            return

        slices = {pid: s.buffers[pid].get_recent_slice(LIVE_SLICE_S) for pid in synced if pid in s.buffers}
        room, phones = s.room, dict(synced)
        floor_rms = {
            pid: float(det._floor)
            for pid, det in s.onset_detectors.items()
            if getattr(det, "_floor", None) is not None
        }
        sus_pre_rms = dict(self._sustained.pre_rms) if self._sustained is not None else {}
        prior_hint = (
            self._sustained.tracker.peek_prediction(SUSTAIN_REEMIT_S, room)
            if (self._sustained is not None and self._sustained.tracker is not None)
            else None
        )

        def snapshot_pre_rms(sub: dict[str, PhoneAudioSlice], t0_s: float) -> dict[str, float]:
            pre_win = engine.extract_aligned_window(sub, t0_s - 0.12, t0_s - 0.02, target_fs=TARGET_FS)
            out_pre: dict[str, float] = {}
            for pid in set(sub) | set(floor_rms):
                vals: list[float] = []
                if pid in pre_win and len(pre_win[pid]):
                    vals.append(_rms(pre_win[pid]))
                fb = floor_rms.get(pid)
                if fb is not None and fb > 0.0:
                    vals.append(fb)
                if vals:
                    out_pre[pid] = min(vals)
            return out_pre

        def analyze(event_id, t0, kind, seq, pids):
            sub = {p: slices[p] for p in pids if p in slices}
            if kind == "sustained" and seq > 0 and not still_sustained(engine, sub, t0, seq, pre_rms=sus_pre_rms):
                return event_id, t0, kind, seq, None
            if kind is None:
                kind = decide_kind(engine, sub, t0, floor_rms=floor_rms)
                if kind is None:
                    return event_id, t0, kind, seq, None
                if kind == "sustained":  # need the full 0.75 s: re-queue as a sustained job
                    return event_id, t0, kind, -1, snapshot_pre_rms(sub, t0)
            aligned = extract_event(engine, sub, t0, kind, seq, min_phones=self.min_phones)
            if len(aligned) < self.min_phones:
                return event_id, t0, kind, seq, None
            used = {p: phones[p] for p in aligned}
            aligned_loc = aligned
            if kind == "sustained" and seq > 0:
                n_loc = int(round(0.45 * TARGET_FS))
                aligned_loc = {p: (arr[-n_loc:] if len(arr) > n_loc else arr) for p, arr in aligned.items()}
            if kind == "sustained" and seq > 0 and prior_hint is not None:
                try:
                    result = s.localizer.localize(
                        aligned_loc, used, room, kind, fs=TARGET_FS,
                        prior_xy=(prior_hint[0], prior_hint[1]), prior_sigma_m=prior_hint[2],
                    )
                except TypeError:
                    result = s.localizer.localize(aligned_loc, used, room, kind, fs=TARGET_FS)
            else:
                result = s.localizer.localize(aligned_loc, used, room, kind, fs=TARGET_FS)
            if result is None:
                return event_id, t0, kind, seq, None
            cls = s.classifier.classify_swarm(aligned, fs=TARGET_FS)
            return event_id, t0, kind, seq, result.model_copy(update={
                "event_id": event_id, "timestamp_s": time.time(), "update_seq": seq, "kind": kind,
                "sound_class": cls.sound_class, "is_drone": cls.is_drone,
                "class_confidence": cls.confidence, "class_probs": cls.probs,
            })

        for job in jobs:
            event_id, t0, kind, seq, payload_or_result = await asyncio.to_thread(analyze, *job[:5])
            if s.clock_epoch != self._epoch:  # recalibrated meanwhile: drop stale results
                return
            if kind == "sustained":
                if seq == -1:  # decided sustained, emit seq 0 once 0.75 s is buffered
                    self._sustained = _Sustained(
                        event_id, t0, set(job[4]), seq=-1,
                        pre_rms=payload_or_result if isinstance(payload_or_result, dict) else {},
                    )
                    self.grouper.hold(sustained_hold_until(t0, 0))
                    continue
                result = payload_or_result
                if result is None:
                    self._sustained = None
                    continue
                prev_sus = self._sustained
                total_votes = (prev_sus.total_votes if prev_sus is not None else 0) + 1
                drone_votes = (prev_sus.drone_votes if prev_sus is not None else 0) + (
                    1 if result.sound_class == "drone" else 0
                )
                drone_conf = (
                    result.class_confidence
                    if result.sound_class == "drone"
                    else (prev_sus.drone_conf if prev_sus is not None else 0.0)
                )
                drone_probs = (
                    result.class_probs
                    if result.sound_class == "drone"
                    else (prev_sus.drone_probs if prev_sus is not None else None)
                )
                is_sticky_drone = (
                    result.sound_class != "drone"
                    and drone_votes >= 2
                    and drone_votes * 2 >= total_votes
                )
                if seq == 0 or prev_sus is None or prev_sus.tracker is None:
                    tracker = MovingTracker2D(result.x_m, result.y_m)
                    ema_xy = tracker.position()
                elif is_sticky_drone:
                    # Keep the settled drone position rather than drifting on fade-out/handling noise.
                    tracker = prev_sus.tracker
                    ema_xy = tracker.position()
                    result = result.model_copy(
                        update={"x_m": round(ema_xy[0], 4), "y_m": round(ema_xy[1], 4)}
                    )
                else:
                    tracker = prev_sus.tracker
                    n_in = sum(1 for pt in result.pair_tdoas_ms if pt.inlier)
                    fx, fy, _ = tracker.step(
                        SUSTAIN_REEMIT_S,
                        result.x_m,
                        result.y_m,
                        residual_ms=result.residual_ms,
                        confidence=result.confidence,
                        n_inliers=n_in,
                        n_pairs=len(result.pair_tdoas_ms),
                        room=room,
                    )
                    ema_xy = (fx, fy)
                    result = result.model_copy(update={"x_m": round(fx, 4), "y_m": round(fy, 4)})
                if is_sticky_drone:
                    result = result.model_copy(
                        update={
                            "sound_class": "drone",
                            "is_drone": True,
                            "class_confidence": drone_conf,
                            "class_probs": drone_probs or result.class_probs,
                        }
                    )
                self._sustained = _Sustained(
                    event_id, t0, set(result.phones_used), seq,
                    pre_rms=prev_sus.pre_rms if prev_sus is not None else sus_pre_rms,
                    ema_xy=ema_xy,
                    tracker=tracker,
                    drone_votes=drone_votes,
                    total_votes=total_votes,
                    drone_conf=drone_conf,
                    drone_probs=drone_probs,
                )
                self.grouper.hold(sustained_hold_until(t0, seq))
                if seq + 1 >= SUSTAIN_MAX_UPDATES:
                    self._sustained = None
            else:
                result = payload_or_result
            if result is not None:
                s.record_detection(result)
                payload = {"type": "detection", **result.model_dump(mode="json")}
                broadcast_s = time.monotonic()
                self.broadcast(payload)
                recording = s.recording
                onset_s = job[5]
                if (kind == "transient" and recording is not None and recording.stopped_at_s is None
                        and onset_s is not None and onset_s >= recording.started_monotonic_s):
                    recording.event_timings.append({
                        "event_id": event_id,
                        "sound_class": result.sound_class,
                        "onset_offset_s": onset_s - recording.started_monotonic_s,
                        "broadcast_offset_s": broadcast_s - recording.started_monotonic_s,
                    })
