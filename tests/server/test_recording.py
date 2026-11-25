from contextlib import ExitStack
import math
import subprocess
import sys

import numpy as np
from fastapi.testclient import TestClient

from backend.dsp.chirp import generate_lfm_chirp
from backend.dsp.replay import load_recording_npz
from backend.server.app import create_app
from backend.server.audio_buffer import BUFFER_DURATION_S
from backend.server.session import Session
from shared.protocol import FLAG_CLOCK_JUMP, encode_audio_frame, speed_of_sound


def send_hello(socket, pid, fs=8000, x=0.5, y=0.5):
    socket.send_json({
        "type": "phone_hello", "phone_id": pid, "label": pid.upper(),
        "sample_rate": fs, "x_m": x, "y_m": y,
    })
    assert socket.receive_json()["type"] == "phone_status"


def test_recording_lifecycle_empty_and_no_overwrite(tmp_path):
    with TestClient(create_app(recordings_dir=tmp_path)) as client:
        assert client.post("/api/record/stop").status_code == 409
        assert client.post("/api/record/start", json={"name": "../escape"}).status_code == 422
        assert client.post("/api/record/start").status_code == 200
        assert client.post("/api/record/start", json={"name": "other"}).status_code == 409
        assert client.post("/api/record/stop").status_code == 409
        assert list(tmp_path.iterdir()) == []

        existing = tmp_path / "spike-01.npz"
        existing.write_bytes(b"keep")
        assert client.post("/api/record/start").status_code == 409
        assert existing.read_bytes() == b"keep"


def test_recording_archives_clock_epochs_and_ignores_retransmissions(tmp_path):
    with TestClient(create_app(recordings_dir=tmp_path)) as client:
        assert client.post("/api/record/start", json={"name": "epochs.npz"}).status_code == 200
        with client.websocket_connect("/ws/phone/nw") as phone:
            send_hello(phone, "nw")
            first = encode_audio_frame(np.full(8, 0.25), 0, 8000, 100)
            phone.send_bytes(first)
            phone.send_bytes(first)
            phone.send_bytes(encode_audio_frame(
                np.full(4, 0.5), 0, 8000, 2, flags=FLAG_CLOCK_JUMP,
            ))
            assert phone.receive_json()["state"] == "clock_jump"
            phone.send_bytes(encode_audio_frame(np.full(16, 0.125), 1, 48000, 120))
            assert phone.receive_json()["state"] == "clock_jump"
        with client.websocket_connect("/ws/phone/nw") as phone:
            send_hello(phone, "nw")
            phone.send_bytes(encode_audio_frame(np.full(2, -0.25), 0, 8000, 200))
            send_hello(phone, "nw")

        result = client.post("/api/record/stop")
        assert result.status_code == 200, result.text
        recording = load_recording_npz(tmp_path / "epochs.npz")
        assert list(recording.slices) == ["nw"]
        assert recording.slices["nw"].start_sample_index == 100
        assert len(recording.slices["nw"].samples_f32) == 8
        with np.load(recording.path, allow_pickle=False) as archive:
            segments = recording.meta["segments"]["nw"]
            assert [s["start_index"] for s in segments] == [100, 2, 120, 200]
            np.testing.assert_array_equal(archive[segments[1]["samples_key"]], 0.5)
            np.testing.assert_array_equal(archive[segments[2]["samples_key"]], 0.125)
            np.testing.assert_array_equal(archive[segments[3]["samples_key"]], -0.25)


def test_gate0_cli_replays_three_chirps_from_capture_longer_than_ring(tmp_path):
    session = Session()
    capture_s = BUFFER_DURATION_S + 5
    gap_size = 2048
    chirps_s = [0.8, 16.0, capture_s - 2.0]
    specs = [("iphone", 44100, 0.5, -0.21, 38.0), ("android", 48000, 7.5, 0.145, -26.0)]
    with TestClient(create_app(lambda: session, recordings_dir=tmp_path)) as client, ExitStack() as stack:
        assert client.post("/api/record/start").status_code == 200
        for pid, fs, x, offset, drift in specs:
            phone = stack.enter_context(client.websocket_connect(f"/ws/phone/{pid}"))
            send_hello(phone, pid, fs=fs, x=x)
            chirp = generate_lfm_chirp(48000)
            source = np.zeros((capture_s + 1)*48000)
            for emission in chirps_s:
                distance_m = math.hypot(x - session.room.speaker_x_m, 0.5 - session.room.speaker_y_m)
                sample = round((emission + distance_m / speed_of_sound()) * 48000)
                source[sample:sample + len(chirp)] = chirp
            local_time = offset + np.arange(capture_s * fs) / (fs * (1 + drift * 1e-6))
            signal = np.interp(local_time * 48000, np.arange(source.size), source).astype(np.float32)
            for seq, start in enumerate(range(0, len(signal), gap_size)):
                if seq == 80:
                    continue
                phone.send_bytes(encode_audio_frame(signal[start:start + gap_size], seq, fs, 10_000 + start))
            send_hello(phone, pid, fs=fs, x=x)
            assert session.buffers[pid].get_recent_slice(99).samples_f32.size <= BUFFER_DURATION_S * fs < capture_s * fs

        response = client.post("/api/record/stop")
        assert response.status_code == 200, response.text

    recording = load_recording_npz(tmp_path / "spike-01.npz")
    for pid, fs, *_ in specs:
        audio = recording.slices[pid]
        assert audio.start_sample_index == 10_000
        assert audio.sample_rate == fs
        assert audio.samples_f32.size == capture_s * fs
        assert int((~audio.valid_mask).sum()) == gap_size
        np.testing.assert_array_equal(audio.samples_f32[80*gap_size:81*gap_size], 0)

    result = subprocess.run(
        [sys.executable, "-m", "backend.dsp.replay", str(recording.path)],
        capture_output=True, text=True, check=True, timeout=20,
    )
    assert "Paired emissions: 3" in result.stdout
    assert "Gate (<1.000 ms, >=2 phones): PASS" in result.stdout


def test_failed_publish_keeps_capture_for_retry(tmp_path):
    with TestClient(create_app(recordings_dir=tmp_path)) as client:
        assert client.post("/api/record/start").status_code == 200
        with client.websocket_connect("/ws/phone/nw") as phone:
            send_hello(phone, "nw")
            phone.send_bytes(encode_audio_frame(np.full(10, 0.25), 0, 8000, 100))
            send_hello(phone, "nw")

        path = tmp_path / "spike-01.npz"
        path.write_bytes(b"existing")
        assert client.post("/api/record/stop").status_code == 409
        assert path.read_bytes() == b"existing"
        path.unlink()

        assert client.post("/api/record/stop").status_code == 200
        assert len(load_recording_npz(path).slices["nw"].samples_f32) == 10
        assert set(tmp_path.iterdir()) == {path}


def test_record_stop_accepts_optional_name_query_param(tmp_path):
    with TestClient(create_app(recordings_dir=tmp_path)) as client:
        assert client.post("/api/record/start").status_code == 200
        with client.websocket_connect("/ws/phone/nw") as phone:
            send_hello(phone, "nw")
            phone.send_bytes(encode_audio_frame(np.full(10, 0.25), 0, 8000, 100))
            send_hello(phone, "nw")

        assert client.post("/api/record/stop?name=../escape").status_code == 422
        resp = client.post("/api/record/stop?name=custom-session")
        assert resp.status_code == 200, resp.text
        expected = tmp_path / "custom-session.npz"
        assert expected.exists()
        assert len(load_recording_npz(expected).slices["nw"].samples_f32) == 10
        assert set(tmp_path.iterdir()) == {expected}
