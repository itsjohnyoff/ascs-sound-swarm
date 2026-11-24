"""End-to-end integration tests driving Denis's real Server (Session, EventLoop, App) with Person 2's real DSP."""

from __future__ import annotations

import asyncio
import math

import numpy as np
from fastapi.testclient import TestClient
from scipy.ndimage import map_coordinates

from backend.dsp.chirp import generate_dual_chirp, generate_lfm_chirp
from backend.dsp.pipeline import LocalizationPipeline, SyncEngine
from backend.server.app import create_app
from backend.server.events import EventLoop
from backend.server.session import Session
from shared.protocol import (
    FLAG_CLOCK_JUMP,
    RoomConfig,
    decode_audio_frame,
    encode_audio_frame,
    speed_of_sound,
)


def _receive_type(ws, expected: str):
    for _ in range(40):
        msg = ws.receive_json()
        if msg["type"] == expected:
            return msg
    raise AssertionError(f"Expected WebSocket message type {expected!r} not received")


def _stream_slice_as_binary_frames(
    session: Session,
    pid: str,
    samples: np.ndarray,
    fs: int,
    start_sample_index: int,
    seq_start: int,
    chunk_size: int = 2048,
) -> tuple[int, int]:
    """Encode float32 audio into 32-byte ASCS binary frames, decode, and ingest into Session."""
    seq = seq_start
    idx = 0
    n = len(samples)
    while idx < n:
        chunk = samples[idx : idx + chunk_size]
        raw = encode_audio_frame(
            chunk,
            seq=seq,
            sample_rate=fs,
            sample_index=start_sample_index + idx,
            client_time_ms=float((start_sample_index + idx) / fs * 1000.0),
        )
        frame = decode_audio_frame(raw)
        # Session.ingest returns True only when sample rate changes or FLAG_CLOCK_JUMP is set
        assert session.ingest(pid, frame) is False
        seq += 1
        idx += len(chunk)
    return start_sample_index + n, seq


def test_session_discovers_real_dsp_backends() -> None:
    session = Session()
    assert isinstance(session.sync_engine, SyncEngine)
    assert isinstance(session.localizer, LocalizationPipeline)


def test_end_to_end_binary_frames_dual_chirp_sync_and_clap_localization() -> None:
    async def run() -> None:
        session = Session()
        assert isinstance(session.sync_engine, SyncEngine)
        assert isinstance(session.localizer, LocalizationPipeline)

        room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
        session.update_room(room)
        c = speed_of_sound(room.temperature_c)

        specs = [
            ("nw", 0.5, 0.5, 48000, -0.180, +35.0),
            ("ne", 7.5, 0.5, 44100, +0.120, -30.0),
            ("sw", 0.5, 5.5, 48000, +0.210, +20.0),
            ("se", 7.5, 5.5, 44100, -0.090, -40.0),
        ]

        fs_sim = 48000
        total_global_s = 8.4
        dual = generate_dual_chirp(fs_sim)
        rng = np.random.default_rng(707)

        # Clap at (3.1, 3.8) at t_global = 7.55 s
        clap_pos = (3.1, 3.8)
        t_clap_global = 7.55
        burst_len = int(round(0.025 * fs_sim))
        t_b = np.arange(burst_len, dtype=np.float64) / fs_sim
        clap_burst = (0.75 * np.exp(-t_b / 0.0045) * rng.normal(0.0, 1.0, size=burst_len)).astype(np.float32)

        phone_streams: dict[str, tuple[np.ndarray, int, int, float]] = {}
        ambient_std = 0.0015

        for idx, (pid, x_m, y_m, fs_phone, offset_s, drift_ppm) in enumerate(specs):
            g_sig = np.zeros(int((total_global_s + 1.0) * fs_sim), dtype=np.float32)

            # Dual chirp emitted at t = 0.60 s (plus +7 ms 0.65x wall echo)
            tau_spk = math.hypot(x_m - room.speaker_x_m, y_m - room.speaker_y_m) / c
            i_c = int(round((0.60 + tau_spk) * fs_sim))
            i_echo = i_c + int(round(0.007 * fs_sim))
            g_sig[i_c : i_c + len(dual)] += dual
            g_sig[i_echo : i_echo + len(dual)] += 0.65 * dual

            # Clap arrival + 0.30x wall echo
            tau_clap = math.hypot(x_m - clap_pos[0], y_m - clap_pos[1]) / c
            i_cl = int(round((t_clap_global + tau_clap) * fs_sim))
            g_sig[i_cl : i_cl + len(clap_burst)] += clap_burst
            g_sig[i_cl + int(round(0.008 * fs_sim)) : i_cl + int(round(0.008 * fs_sim)) + len(clap_burst)] += (
                0.30 * clap_burst
            )

            fs_true = fs_phone * (1.0 + drift_ppm * 1e-6)
            # Sample up to global t = total_global_s from each phone's realistic start offset
            n_phone = int(round((total_global_s - offset_s) * fs_true))
            t_glob = offset_s + np.arange(n_phone, dtype=np.float64) / fs_true
            rec = map_coordinates(g_sig.astype(np.float64), [t_glob * fs_sim], order=3, mode="constant", cval=0.0).astype(
                np.float32
            )
            rec += rng.normal(0.0, ambient_std, size=len(rec)).astype(np.float32)

            session.connect(pid)
            session.set_phone(pid, label=pid.upper(), x_m=x_m, y_m=y_m)
            phone_streams[pid] = (rec, fs_phone, 10_000 + idx * 4_000, offset_s)

        # 1. Stream pre-arm lead-in up to global t = 0.40 s
        cursors: dict[str, tuple[int, int]] = {}
        for pid, (rec, fs_phone, base_idx, offset_s) in phone_streams.items():
            n_lead = max(2048, int(round((0.40 - offset_s) * fs_phone)))
            _, next_seq = _stream_slice_as_binary_frames(
                session, pid, rec[:n_lead], fs_phone, base_idx, seq_start=0
            )
            cursors[pid] = (n_lead, next_seq)

        # 2. Arm dual-chirp calibration, stream through global t = 6.60 s, and solve via Session.solve
        armed = session.arm("lfm_dual")
        for pid, (rec, fs_phone, base_idx, offset_s) in phone_streams.items():
            pos, seq = cursors[pid]
            n_cal_end = int(round((6.60 - offset_s) * fs_phone))
            _, next_seq = _stream_slice_as_binary_frames(
                session, pid, rec[pos:n_cal_end], fs_phone, base_idx + pos, seq_start=seq
            )
            cursors[pid] = (n_cal_end, next_seq)

        cal = await session.solve(armed.calibration_id, "lfm_dual", window_s=7.0)
        assert cal.success is True
        assert cal.synced_count == 4
        assert cal.max_residual_ms is not None and cal.max_residual_ms < 0.25
        assert all(session.phones[pid].sync_state == "synced" for pid, *_ in specs)

        # 3. Stream post-calibration frames through global t = 8.4 s containing the clap and run the real EventLoop
        sent: list[dict] = []
        processor = EventLoop(session, sent.append)
        for pid, (rec, fs_phone, base_idx, _) in phone_streams.items():
            pos, seq = cursors[pid]
            _stream_slice_as_binary_frames(
                session, pid, rec[pos:], fs_phone, base_idx + pos, seq_start=seq
            )

        await processor.process_once()
        sent = [m for m in sent if m["type"] == "detection"]
        assert len(sent) == 1
        det = sent[0]
        assert det["type"] == "detection"
        assert set(det["phones_used"]) == {"nw", "ne", "sw", "se"}
        assert det["heatmap"]["nx"] == 32 and det["heatmap"]["ny"] == 24

        err_m = math.hypot(det["x_m"] - clap_pos[0], det["y_m"] - clap_pos[1])
        print(
            f"\nE2E Server+DSP integration: cal_max_res={cal.max_residual_ms:.4f} ms, "
            f"gt={clap_pos}, est=({det['x_m']:.3f}, {det['y_m']:.3f}), err={err_m:.3f} m"
        )
        assert err_m < 0.15

        # 4. Log detection into JudgeEval
        trial = session.judge.add_trial("Judge Pos 1", clap_pos[0], clap_pos[1], "clap", session.detections[det["event_id"]])
        assert trial.missed is False
        assert trial.error_m is not None and trial.error_m < 0.15

    asyncio.run(run())


def test_end_to_end_fastapi_ws_calibration_and_clock_jump() -> None:
    fs = 48000
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)
    chirp = generate_lfm_chirp(fs)

    with TestClient(create_app()) as client:
        with (
            client.websocket_connect("/ws/dashboard") as dashboard,
            client.websocket_connect("/ws/phone/nw") as ws_nw,
            client.websocket_connect("/ws/phone/ne") as ws_ne,
        ):
            _receive_type(dashboard, "session_state")

            for sock, pid, x_m, y_m in ((ws_nw, "nw", 0.5, 0.5), (ws_ne, "ne", 7.5, 0.5)):
                sock.send_json(
                    {
                        "type": "phone_hello",
                        "phone_id": pid,
                        "label": pid.upper(),
                        "x_m": x_m,
                        "y_m": y_m,
                        "sample_rate": fs,
                        "device_info": "integration test",
                    }
                )
                assert _receive_type(sock, "phone_status")["state"] == "unsynced"
                sock.send_bytes(encode_audio_frame(np.zeros(2048, dtype=np.float32), seq=0, sample_rate=fs, sample_index=0))

            arm = client.post("/api/calibrate/arm", json={"mode": "lfm_single"}).json()

            # Stream 0.8 s containing the LFM chirp with exact acoustic flight times
            n_cal = int(round(0.8 * fs))
            for sock, pid, x_m, y_m in ((ws_nw, "nw", 0.5, 0.5), (ws_ne, "ne", 7.5, 0.5)):
                tau_prop = math.hypot(x_m - room.speaker_x_m, y_m - room.speaker_y_m) / c
                buf = np.zeros(n_cal, dtype=np.float32)
                s0 = int(round((0.15 + tau_prop) * fs))
                buf[s0 : s0 + len(chirp)] = chirp
                seq = 1
                idx = 0
                while idx < len(buf):
                    chunk = buf[idx : idx + 2048]
                    sock.send_bytes(encode_audio_frame(chunk, seq=seq, sample_rate=fs, sample_index=2048 + idx))
                    seq += 1
                    idx += len(chunk)

            solve_resp = client.post(
                "/api/calibrate/solve",
                json={"calibration_id": arm["calibration_id"], "mode": "lfm_single", "window_s": 2.0},
            )
            assert solve_resp.status_code == 200
            body = solve_resp.json()
            assert body["success"] is True
            assert body["synced_count"] == 2
            assert _receive_type(dashboard, "calibration_result")["synced_count"] == 2
            assert _receive_type(ws_nw, "phone_status")["synced"] is True
            assert _receive_type(ws_ne, "phone_status")["synced"] is True
            assert client.get("/api/health").json()["synced_phones"] == 2

            # Clock jump on the reference phone: v2 re-references, but one phone alone has no
            # shared timeline, so nothing stays synced (CONTRACTS 2).
            ws_nw.send_bytes(
                encode_audio_frame(
                    np.zeros(2048, dtype=np.float32),
                    seq=0,
                    sample_rate=fs,
                    sample_index=0,
                    flags=FLAG_CLOCK_JUMP,
                )
            )
            status = _receive_type(ws_nw, "phone_status")
            assert status["synced"] is False and status["state"] == "clock_jump"
            assert client.get("/api/health").json()["synced_phones"] == 0
            assert client.get("/api/session").json()["phones"][1]["sync_state"] == "unsynced"
