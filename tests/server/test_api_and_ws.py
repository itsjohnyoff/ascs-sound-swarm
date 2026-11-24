import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.server.app import create_app
from backend.server.session import Session
from shared.protocol import (
    FLAG_CLOCK_JUMP,
    CalibrationResult,
    LocalizationResult,
    PhoneSyncReport,
    encode_audio_frame,
)


def receive_type(socket, expected):
    for _ in range(30):
        message = socket.receive_json()
        if message["type"] == expected:
            return message
    pytest.fail(f"No {expected} message received")


def hello(socket, phone_id="nw", x=0.5, y=0.5):
    socket.send_json({
        "type": "phone_hello", "phone_id": phone_id, "label": phone_id.upper(),
        "x_m": x, "y_m": y, "sample_rate": 8000, "device_info": "test phone",
    })
    return receive_type(socket, "phone_status")


def send_audio(socket, seq=0, start=0, flags=0):
    socket.send_bytes(encode_audio_frame(
        np.full(800, 0.25, dtype=np.float32), seq, 8000, start, flags=flags,
    ))


def test_room_update_broadcast_and_join_qr(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://demo.example/")
    (tmp_path / "phone.html").write_text("<title>Sound Swarm</title>")
    with TestClient(create_app(frontend_dir=tmp_path)) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["phones_connected"] == health["synced_phones"] == 0

        with client.websocket_connect("/ws/dashboard") as dashboard:
            initial = receive_type(dashboard, "session_state")
            assert initial["room"]["width_m"] == 8.0
            assert initial["gdop_warning"].startswith("Need at least 3 phones")
            response = client.put("/api/room", json={
                "width_m": 9, "height_m": 7, "temperature_c": 23,
                "speaker_x_m": 4, "speaker_y_m": 0.5,
            })
            assert response.json()["width_m"] == 9
            assert receive_type(dashboard, "session_state")["room"] == response.json()

        assert client.get("/phone.html").text == "<title>Sound Swarm</title>"
        join = client.get("/api/join-info").json()
        assert join["join_url"] == "https://demo.example/phone.html"
        assert join["qr_svg"].startswith("<svg")
        assert "http://localhost:5173" == client.get(
            "/api/health", headers={"Origin": "http://localhost:5173"},
        ).headers["access-control-allow-origin"]
        assert "access-control-allow-origin" not in client.get(
            "/api/health", headers={"Origin": "https://foreign.example"},
        ).headers


def test_binary_ingestion_telemetry_and_disconnect():
    session = Session()
    with TestClient(create_app(lambda: session)) as client:
        with client.websocket_connect("/ws/dashboard") as dashboard:
            with client.websocket_connect("/ws/phone/nw") as phone:
                assert hello(phone)["state"] == "unsynced"
                phone.send_text("{")
                phone.send_bytes(b"malformed")
                send_audio(phone)
                send_audio(phone, seq=2, start=1600)
                for _ in range(10):
                    telemetry = receive_type(dashboard, "telemetry")
                    if telemetry["phones"] and telemetry["phones"][0]["dropped_frames"] == 1:
                        break
                assert telemetry["phones"] == [{
                    "phone_id": "nw", "rms": pytest.approx(0.25, abs=1e-4),
                    "connected": True, "sync_state": "unsynced", "dropped_frames": 1,
                }]
                audio = session.buffers["nw"].get_recent_slice(1)
                assert audio.start_sample_index == 0
                np.testing.assert_array_equal(audio.valid_mask[800:1600], False)
                assert client.delete("/api/phones/nw").status_code == 409
                assert client.put("/api/phones/nw", json={"x_m": 2, "y_m": 1}).json()["label"] == "NW"
                assert client.put("/api/phones/nw", json={"x_m": -1, "y_m": 1}).status_code == 422

            assert client.get("/api/health").json()["phones_connected"] == 0
            assert client.delete("/api/phones/nw").status_code == 204
            assert client.get("/api/session").json()["phones"] == []


class TestSyncEngine:
    __test__ = False

    def __init__(self):
        self.synced = set()

    def mark_clock_jump(self, phone_id):
        self.synced.discard(phone_id)

    def is_synced(self, phone_id):
        return phone_id in self.synced

    def calibrate_from_buffers(self, buffers, phones, room, mode):
        assert set(phones) == set(buffers)
        self.synced = {pid for pid, audio in buffers.items() if audio.start_sample_index == 800}
        return CalibrationResult(
            mode=mode, success=len(self.synced) >= 2, synced_count=len(self.synced),
            median_residual_ms=0.2, max_residual_ms=0.2,
            per_phone={pid: PhoneSyncReport(
                synced=pid in self.synced, residual_ms=0.2, drift_ppm=3, chirp_snr_db=24,
            ) for pid in phones},
        )


def test_armed_solve_broadcasts_and_clock_jump_invalidates():
    session = Session(sync_engine=TestSyncEngine())
    with TestClient(create_app(lambda: session)) as client:
        with (
            client.websocket_connect("/ws/dashboard") as dashboard,
            client.websocket_connect("/ws/phone/nw") as first,
            client.websocket_connect("/ws/phone/ne") as second,
        ):
            hello(first)
            hello(second, "ne", x=7.5)
            for socket in (first, second):
                send_audio(socket)
            for _ in range(10):
                snapshot = receive_type(dashboard, "telemetry")
                if len([p for p in snapshot["phones"] if p["rms"] > 0]) == 2:
                    break
            arm = client.post("/api/calibrate/arm", json={"mode": "lfm_single"}).json()
            request = {"calibration_id": arm["calibration_id"], "mode": "lfm_single", "window_s": 2}
            for socket in (first, second):
                send_audio(socket, 1, 800)
                hello(socket, "nw" if socket is first else "ne", x=0.5 if socket is first else 7.5)
            response = client.post("/api/calibrate/solve", json=request)
            assert response.status_code == 200
            result = receive_type(dashboard, "calibration_result")
            assert result == response.json()
            assert result["success"] and result["synced_count"] == 2
            assert receive_type(first, "phone_status")["synced"] is True
            assert client.get("/api/health").json()["synced_phones"] == 2
            assert client.post("/api/calibrate/solve", json=request).json() == result

            send_audio(first, 0, 0, flags=FLAG_CLOCK_JUMP)
            status = receive_type(first, "phone_status")
            assert status["synced"] is False and status["state"] == "clock_jump"
            assert not session.sync_engine.is_synced("nw")
            assert client.get("/api/health").json()["synced_phones"] == 1


def test_clock_jump_during_solve_cannot_publish_stale_models():
    started, release = threading.Event(), threading.Event()

    class BlockingSyncEngine(TestSyncEngine):
        def calibrate_from_buffers(self, buffers, phones, room, mode):
            started.set()
            assert release.wait(5)
            return super().calibrate_from_buffers(buffers, phones, room, mode)

    session = Session(sync_engine=BlockingSyncEngine())
    with TestClient(create_app(lambda: session)) as client:
        with client.websocket_connect("/ws/phone/nw") as phone:
            hello(phone)
            arm = client.post("/api/calibrate/arm", json={"mode": "lfm_single"}).json()
            send_audio(phone, start=800)
            hello(phone)
            with ThreadPoolExecutor(max_workers=1) as pool:
                solve = pool.submit(client.post, "/api/calibrate/solve", json={
                    "calibration_id": arm["calibration_id"], "mode": "lfm_single",
                })
                try:
                    assert started.wait(5)
                    send_audio(phone, flags=FLAG_CLOCK_JUMP)
                    assert receive_type(phone, "phone_status")["state"] == "clock_jump"
                finally:
                    release.set()
                assert solve.result(timeout=5).status_code == 409
            assert not session.sync_engine.is_synced("nw")


def test_trials_match_retained_detections_or_explicit_misses():
    session = Session()
    for i in range(101):
        session.record_detection(LocalizationResult(
            event_id=f"evt-{i}", timestamp_s=1_700_000_000.0,
            x_m=3, y_m=4, uncertainty_m=0.1, residual_ms=0.2,
            confidence=0.9, phones_used=["nw", "ne", "sw"], sound_class="clap",
        ))
    with TestClient(create_app(lambda: session)) as client:
        trial = {"label": "Position 1", "gt_x_m": 0, "gt_y_m": 0, "expected_class": "clap"}
        assert client.post("/api/eval/trials", json={**trial, "event_id": "evt-0"}).status_code == 404
        assert client.post("/api/eval/trials", json={**trial, "event_id": "evt-1"}).json()["error_m"] == 5
        assert client.post("/api/eval/trials", json={**trial, "event_id": None}).json()["missed"]
        summary = client.get("/api/eval/trials").json()["summary"]
        assert (summary["total"], summary["detected"], summary["missed"]) == (2, 1, 1)
        export = client.get("/api/eval/export.csv")
        assert export.headers["content-disposition"].endswith('"judge-eval.csv"')
        assert export.content.count(b"\r\n") == 3
        assert client.delete("/api/eval/trials").status_code == 204
        assert client.get("/api/eval/trials").json()["trials"] == []
