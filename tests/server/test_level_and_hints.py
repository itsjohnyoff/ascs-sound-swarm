"""Loudness check before calibrating, and instructions pushed to a phone's screen."""

import asyncio

import numpy as np
from fastapi.testclient import TestClient

from backend.dsp.chirp import generate_lfm_chirp
from backend.server.app import create_app
from backend.server.session import Session, level_report
from shared.protocol import DecodedAudioFrame, encode_audio_frame

FS = 48000


def _window(gain: float, noise: float = 0.002, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(2 * FS) * noise
    chirp = generate_lfm_chirp(FS)
    x[FS:FS + len(chirp)] += gain * chirp
    return np.clip(x, -1, 1).astype(np.float32)


def test_level_verdicts():
    assert level_report(_window(0.3), FS)["verdict"] == "ok"
    assert level_report(_window(0.0005), FS)["verdict"] == "quiet"  # heard, but under the 28 dB margin
    assert level_report(_window(0.0), FS)["verdict"] == "silent"
    assert level_report(_window(2.0), FS)["verdict"] == "clipping"
    assert level_report(np.zeros(0, np.float32), FS)["verdict"] == "silent"


def test_session_level_check_reads_only_audio_after_arming():
    session = Session()
    for pid in ("loud", "far"):
        session.connect(pid)
    level_id = session.arm_level()
    for pid, gain in (("loud", 0.3), ("far", 0.0)):
        x = _window(gain)
        session.ingest(pid, DecodedAudioFrame(0, len(x), 0, FS, 0, 0.0, x))
    result = asyncio.run(session.solve_level(level_id))
    assert result["ok"] == 1 and result["total"] == 2
    assert result["per_phone"]["loud"]["verdict"] == "ok"
    assert result["per_phone"]["far"]["verdict"] == "silent"


def test_level_solve_pushes_a_hint_and_dashboard_can_send_one():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/phone/a") as phone:
            phone.send_bytes(encode_audio_frame(np.zeros(800, np.float32), 0, FS, 0))
            level_id = client.post("/api/level/arm").json()["level_id"]
            assert client.post("/api/level/solve", json={"level_id": level_id}).json()["per_phone"]["a"]["verdict"] == "silent"
            while (m := phone.receive_json())["type"] != "phone_hint":
                pass
            assert m["title"] == "I can't hear the laptop"

            ok = client.post("/api/phones/a/hint", json={"tone": "warn", "title": "Clap right next to me"})
            assert ok.status_code == 200
            while (m := phone.receive_json())["type"] != "phone_hint":
                pass
            assert m["title"] == "Clap right next to me"
        assert client.post("/api/phones/a/hint", json={"title": "x"}).status_code == 404
