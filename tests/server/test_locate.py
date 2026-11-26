"""Self-localization through the server: session solve on simulated audio, and the emit schedule over WS."""

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.dsp import ranging
from backend.server.app import create_app
from backend.server.session import CalibrationConflict, Session
from shared.protocol import DecodedAudioFrame, encode_audio_frame
from sim.room import CENTRED_ROOM, locate_scenario

FRAME = 2048


def _session_with(rec, rough: dict[str, tuple[float, float]]) -> tuple[Session, dict]:
    meta = json.loads(str(rec["meta"]))
    session = Session()
    session.update_room(CENTRED_ROOM)
    for pid in sorted(meta["phones"]):
        session.connect(pid)
        session.set_phone(pid, x_m=rough[pid][0], y_m=rough[pid][1])
    return session, meta


def _feed(session: Session, rec, meta) -> None:
    for pid in sorted(meta["phones"]):
        pcm, fs = rec[f"{pid}_pcm"], int(rec[f"{pid}_fs"])
        start = meta["truth"]["phones"][pid]["start_index"]
        for seq, k in enumerate(range(0, len(pcm), FRAME)):
            chunk = pcm[k:k + FRAME]
            session.ingest(pid, DecodedAudioFrame(0, len(chunk), seq, fs, start + k, 0.0, chunk))


def test_session_locates_phones_from_rough_guesses_and_flags_a_move():
    rec = locate_scenario(seed=2)
    truth = {p: (v["x_m"], v["y_m"]) for p, v in json.loads(str(rec["meta"]))["phones"].items()}
    rough = {p: (x + 0.4, y - 0.3) for p, (x, y) in truth.items()}  # roughly right, as if dragged by eye
    session, meta = _session_with(rec, rough)

    run = session.arm_locate()
    _feed(session, rec, meta)
    result = asyncio.run(session.solve_locate(run.locate_id))

    assert result["success"] and result["applied"] and result["anchored"]
    errors = [np.hypot(session.phones[p].x_m - x, session.phones[p].y_m - y) for p, (x, y) in truth.items()]
    assert np.median(errors) < 0.05
    assert all(p["status"] == "ok" and p["moved_m"] is None for p in result["phones"].values())

    # Same audio again, but the map says p3 is 40 cm away from where it was found: a check run flags it.
    session2, meta2 = _session_with(rec, {p: (session.phones[p].x_m, session.phones[p].y_m) for p in truth})
    session2.located = {p: (session.phones[p].x_m, session.phones[p].y_m) for p in truth}
    session2.located["p3"] = (session2.located["p3"][0] - 0.4, session2.located["p3"][1])
    check = session2.arm_locate(apply=False)
    _feed(session2, rec, meta2)
    out = asyncio.run(session2.solve_locate(check.locate_id))
    assert not out["applied"]
    assert out["phones"]["p3"]["moved"] and not out["phones"]["p1"]["moved"]


def test_locate_needs_three_phones():
    session = Session()
    for pid in ("a", "b"):
        session.connect(pid)
    with pytest.raises(CalibrationConflict, match="at least 3"):
        session.arm_locate()


def test_turn_layout_rotates_about_the_speaker_and_mirrors():
    session = Session()
    session.update_room(CENTRED_ROOM)
    session.connect("a")
    session.set_phone("a", x_m=5.0, y_m=3.0)  # 1 m right of the speaker at (4, 3)
    session.turn_layout(90, mirror=False)
    assert (session.phones["a"].x_m, session.phones["a"].y_m) == pytest.approx((4.0, 4.0))
    session.turn_layout(0, mirror=True)
    assert (session.phones["a"].x_m, session.phones["a"].y_m) == pytest.approx((4.0, 4.0))
    session.set_phone("a", x_m=5.0, y_m=3.0)
    session.turn_layout(0, mirror=True)
    assert session.phones["a"].x_m == pytest.approx(3.0)


def test_arm_sends_each_phone_its_emit_slots_in_schedule_order(monkeypatch):
    monkeypatch.setattr(ranging, "SLOT_S", 0.02)
    with TestClient(create_app()) as client:
        sockets = []
        try:
            for pid in ("a", "b", "c"):
                ws = client.websocket_connect(f"/ws/phone/{pid}").__enter__()
                sockets.append(ws)
                ws.send_bytes(encode_audio_frame(np.zeros(800, np.float32), 0, 8000, 0))
            for ws, pid in zip(sockets, ("a", "b", "c")):
                ws.send_json({"type": "phone_hello", "phone_id": pid, "x_m": 1, "y_m": 1, "sample_rate": 8000})
            armed = client.post("/api/locate/arm").json()
            assert armed["order"] == ["a", "b", "c"] and armed["laptop_slots"] == [0, 7]
            for ws, pid in zip(sockets, ("a", "b", "c")):
                slots = []
                while len(slots) < 2:
                    m = ws.receive_json()
                    if m["type"] == "emit_chirp":
                        slots.append(m["slot"])
                expected = [k for k, who in enumerate(ranging.schedule(["a", "b", "c"])) if who == pid]
                assert slots == expected
        finally:
            for ws in sockets:
                ws.__exit__(None, None, None)
