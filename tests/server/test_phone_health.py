"""Phones report round trip and battery; the server echoes pings and shares the numbers with the dashboard."""

import numpy as np
from fastapi.testclient import TestClient

from backend.server.app import create_app
from shared.protocol import encode_audio_frame


def test_ping_is_echoed_and_health_reaches_the_dashboard():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/phone/a") as phone:
            phone.send_bytes(encode_audio_frame(np.zeros(800, np.float32), 0, 8000, 0))
            phone.send_json({"type": "phone_ping", "t": 1234.5, "rtt_ms": 42, "battery": 0.55, "charging": True})
            while (m := phone.receive_json())["type"] != "phone_pong":
                pass
            assert m["t"] == 1234.5
            a = next(p for p in client.get("/api/session").json()["phones"] if p["phone_id"] == "a")
            assert (a["rtt_ms"], a["battery"], a["charging"]) == (42, 0.55, True)

            phone.send_json({"type": "phone_ping", "t": 1, "battery": 7})  # out of range: ignored, no crash
            phone.send_json({"type": "phone_ping", "t": 2})
            while (m := phone.receive_json())["type"] != "phone_pong":
                pass
            assert m["t"] == 2
