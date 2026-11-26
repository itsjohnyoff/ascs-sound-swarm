"""Live event loop (CONTRACTS 4) on simulated phones streamed as real binary frames."""

import asyncio
import json
import math

import numpy as np

from backend.server.events import EventLoop
from backend.server.recording import Recording
from backend.server.session import Session
from shared.protocol import decode_audio_frame, encode_audio_frame
from sim.room import DEFAULT_ROOM, SimEvent, default_phones, simulate, true_time

FRAME = 2048


class Swarm:
    """Feeds each simulated phone into a Session frame by frame, in true-time order."""

    def __init__(self, session, rec):
        self.session, self.rec = session, rec
        meta = json.loads(str(rec["meta"]))
        self.truth = meta["truth"]
        self.pos = {pid: 0 for pid in meta["phones"]}
        self.seq = dict.fromkeys(meta["phones"], 0)
        for pid, p in meta["phones"].items():
            session.connect(pid)
            session.set_phone(pid, x_m=p["x_m"], y_m=p["y_m"], sample_rate=self.truth["phones"][pid]["fs"])

    def stream_until(self, t_true):
        for pid, tp in self.truth["phones"].items():
            if pid not in self.session.buffers:
                continue
            pcm, valid = self.rec[f"{pid}_pcm"], self.rec[f"{pid}_valid"]
            while self.pos[pid] + FRAME <= len(pcm) and true_time(tp, tp["start_index"] + self.pos[pid] + FRAME) <= t_true:
                k = self.pos[pid]
                if valid[k:k + FRAME].all():  # dropped frames are simply never sent
                    raw = encode_audio_frame(pcm[k:k + FRAME], self.seq[pid], tp["fs"], tp["start_index"] + k)
                    self.session.ingest(pid, decode_audio_frame(raw))
                self.seq[pid] += 1
                self.pos[pid] = k + FRAME


def _run(rec, events_until, *, tick=0.1, capture=None):
    async def go():
        session = Session()
        if capture is not None:
            session.recording = Recording(capture, session.room)
        sent = []
        loop = EventLoop(session, sent.append)
        swarm = Swarm(session, rec)
        swarm.stream_until(0.5)
        armed = session.arm("lfm_single")
        swarm.stream_until(1.6)
        cal = await session.solve(armed.calibration_id, "lfm_single", window_s=3.0)
        assert cal.synced_count == 4, cal
        t = 1.6
        while t < events_until:
            t += tick
            swarm.stream_until(t)
            await loop.process_once()
        return session, [m for m in sent if m["type"] == "detection"]
    return asyncio.run(go())


def _rec(events, duration, drop=None, seed=21):
    rng = np.random.default_rng(seed)
    phones = default_phones(rng, 4, snr_db=30.0)
    if drop:
        phones[drop[0]].drop_ranges_s = [drop[1]]
    room = DEFAULT_ROOM
    evs = [SimEvent(1.0, room.speaker_x_m, room.speaker_y_m, "chirp")] + events
    return simulate(room, phones, evs, duration, reflection_gain=0.3, seed=seed)


def test_one_clap_gives_exactly_one_detection_and_chirp_gives_none():
    rec = _rec([SimEvent(3.0, 3.0, 2.5, "clap")], 5.0)
    session, dets = _run(rec, 4.8)
    assert len(dets) == 1
    d = dets[0]
    assert d["kind"] == "transient" and d["update_seq"] == 0
    assert math.hypot(d["x_m"] - 3.0, d["y_m"] - 2.5) < 0.3
    assert d["event_id"] in session.detections


def test_capture_keeps_calibration_and_clap_broadcast_evidence(tmp_path):
    path = tmp_path / "gate1.npz"
    rec = _rec([SimEvent(3.0, 3.0, 2.5, "clap")], 5.0)
    session, dets = _run(rec, 4.8, capture=path)
    recording = session.recording
    try:
        recording.save()
    finally:
        recording.close()
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["meta"].item())

    calibration, = meta["calibrations"]
    assert calibration["success"]
    assert calibration["synced_count"] == 4
    assert set(calibration["per_phone"]) == set(session.phones)
    timing, = meta["event_timings"]
    assert timing["event_id"] == dets[0]["event_id"]
    assert timing["sound_class"] == dets[0]["sound_class"]
    assert 0 <= timing["onset_offset_s"] <= timing["broadcast_offset_s"]
    assert timing["broadcast_offset_s"] > calibration["offset_s"]


def test_drone_is_sustained_and_reemits_with_same_event_id():
    rec = _rec([SimEvent(3.0, 5.0, 3.5, "drone", duration_s=2.0)], 6.5)
    _, dets = _run(rec, 6.3)
    assert dets, "no detection"
    assert {d["event_id"] for d in dets} == {dets[0]["event_id"]}
    assert [d["update_seq"] for d in dets] == list(range(len(dets)))
    assert len(dets) >= 2 and all(d["kind"] == "sustained" for d in dets)
    assert math.hypot(dets[0]["x_m"] - 5.0, dets[0]["y_m"] - 3.5) < 0.3


def test_phone_with_gap_in_window_is_excluded():
    # p3 loses frames around the clap (local time ~3.0-3.3 s); still 3 phones left.
    rec = _rec([SimEvent(3.0, 3.0, 2.5, "clap")], 5.0, drop=(2, (2.7, 3.6)))
    _, dets = _run(rec, 4.8)
    assert len(dets) == 1
    assert "p3" not in dets[0]["phones_used"] and len(dets[0]["phones_used"]) == 3


def test_no_detection_before_calibration():
    async def go():
        session = Session()
        sent = []
        loop = EventLoop(session, sent.append)
        rec = _rec([SimEvent(2.0, 3.0, 2.5, "clap")], 3.0)
        swarm = Swarm(session, rec)
        for k in range(30):
            swarm.stream_until(0.1 * k)
            await loop.process_once()
        return sent
    assert asyncio.run(go()) == []


def test_dropping_reference_phone_rereferences_and_keeps_detecting():
    async def go():
        rec = _rec([SimEvent(3.0, 3.0, 2.5, "clap")], 5.0)
        session = Session()
        sent = []
        loop = EventLoop(session, sent.append)
        swarm = Swarm(session, rec)
        swarm.stream_until(0.5)
        armed = session.arm("lfm_single")
        swarm.stream_until(1.6)
        cal = await session.solve(armed.calibration_id, "lfm_single", window_s=3.0)
        assert cal.synced_count == 4

        ref_id = session.sync_engine._v2_fitter.ref_phone_id
        assert ref_id is not None
        epoch_before = session.clock_epoch

        # Disconnect the reference phone: remaining 3 phones should re-reference onto a new clock epoch
        session.disconnect(ref_id)
        assert session.clock_epoch > epoch_before
        new_ref = session.sync_engine._v2_fitter.ref_phone_id
        assert new_ref is not None and new_ref != ref_id
        assert len(session.synced_phones()) == 3
        # New reference phone should have 0.0 drift relative to itself
        assert session.phones[new_ref].drift_ppm == 0.0

        t = 1.6
        while t < 4.8:
            t += 0.1
            swarm.stream_until(t)
            await loop.process_once()
        return [m for m in sent if m["type"] == "detection"], ref_id

    dets, ref_id = asyncio.run(go())
    assert len(dets) == 1
    assert ref_id not in dets[0]["phones_used"]
    assert len(dets[0]["phones_used"]) == 3
    assert math.hypot(dets[0]["x_m"] - 3.0, dets[0]["y_m"] - 2.5) < 0.35


def test_long_drone_beyond_live_slice_keeps_single_event_id_and_tracks_past_seq_6():
    # Buffer/slice window is 4.0 s; a 5.5 s sustained drone must keep tracking past seq 6 on the same event_id.
    rec = _rec([SimEvent(3.0, 5.0, 3.5, "drone", duration_s=5.5)], 9.5)
    _, dets = _run(rec, 9.2)
    assert dets, "no detection"
    assert {d["event_id"] for d in dets} == {dets[0]["event_id"]}
    assert [d["update_seq"] for d in dets] == list(range(len(dets)))
    assert dets[-1]["update_seq"] >= 7
    assert all(d["kind"] == "sustained" for d in dets)
    assert math.hypot(dets[-1]["x_m"] - 5.0, dets[-1]["y_m"] - 3.5) < 0.3


def test_onset_detector_quiet_mic_floor_not_poisoned():
    from backend.dsp.pipeline import OnsetDetector

    fs = 48000
    det = OnsetDetector()
    # 2 s quiet background at 0.0001 RMS -> floor ~0.0001 (no false onsets)
    rng = np.random.default_rng(11)
    quiet = (rng.normal(0.0, 0.0001, size=2 * fs)).astype(np.float32)
    for k in range(0, len(quiet), FRAME):
        assert det.push(quiet[k:k + FRAME], k, fs) == []
    # 12 s continuous signal dipping around rms ~0.0018 (> 10 dB above 0.0001 floor) must not poison _floor
    sig = (rng.normal(0.0, 0.0018, size=12 * fs)).astype(np.float32)
    onsets = []
    for k in range(0, len(sig), FRAME):
        onsets.extend(det.push(sig[k:k + FRAME], 2 * fs + k, fs))
    assert det._floor is not None and det._floor < 0.0005
    assert len(onsets) >= 20


def test_sustained_drone_keeps_sticky_label_on_fadeout_frame():
    from shared.protocol import ClassificationResult

    class _FadeOutClassifier:
        backend_name = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def classify_swarm(self, aligned, fs: int = 48000) -> ClassificationResult:
            self.calls += 1
            if self.calls <= 3:
                return ClassificationResult(
                    sound_class="drone", is_drone=True, confidence=0.92,
                    probs={"drone": 0.92, "clap": 0.01, "other": 0.07}, backend_name="stub",
                )
            return ClassificationResult(
                sound_class="other", is_drone=False, confidence=0.88,
                probs={"drone": 0.08, "clap": 0.04, "other": 0.88}, backend_name="stub",
            )

    async def go():
        rec = _rec([SimEvent(3.0, 5.0, 3.5, "drone", duration_s=1.5)], 5.5)
        session = Session(classifier=_FadeOutClassifier())
        sent = []
        loop = EventLoop(session, sent.append)
        swarm = Swarm(session, rec)
        swarm.stream_until(0.5)
        armed = session.arm("lfm_single")
        swarm.stream_until(1.6)
        cal = await session.solve(armed.calibration_id, "lfm_single", window_s=3.0)
        assert cal.synced_count == 4
        t = 1.6
        while t < 5.3:
            t += 0.1
            swarm.stream_until(t)
            await loop.process_once()
        return [m for m in sent if m["type"] == "detection"]

    dets = asyncio.run(go())
    assert len(dets) >= 4
    assert all(d["sound_class"] == "drone" and d["is_drone"] is True for d in dets)
    # Trailing fade-out frames preserve the last confident drone position instead of jumping.
    assert (dets[-1]["x_m"], dets[-1]["y_m"]) == (dets[2]["x_m"], dets[2]["y_m"])
