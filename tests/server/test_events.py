"""Live event loop (CONTRACTS 4) on simulated phones streamed as real binary frames."""

import asyncio
import json
import math

import numpy as np

from backend.server.events import EventLoop
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
            pcm, valid = self.rec[f"{pid}_pcm"], self.rec[f"{pid}_valid"]
            while self.pos[pid] + FRAME <= len(pcm) and true_time(tp, tp["start_index"] + self.pos[pid] + FRAME) <= t_true:
                k = self.pos[pid]
                if valid[k:k + FRAME].all():  # dropped frames are simply never sent
                    raw = encode_audio_frame(pcm[k:k + FRAME], self.seq[pid], tp["fs"], tp["start_index"] + k)
                    self.session.ingest(pid, decode_audio_frame(raw))
                self.seq[pid] += 1
                self.pos[pid] = k + FRAME


def _run(rec, events_until, *, tick=0.1):
    async def go():
        session = Session()
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
