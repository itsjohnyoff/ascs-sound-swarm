from fastapi.testclient import TestClient

from backend.server.app import create_app
from backend.server.session import Session, compute_gdop_warning
from shared.protocol import PhoneConfig


def test_gdop_warning_for_collinear_or_too_few_phones():
    phones = [PhoneConfig(phone_id=p, x_m=x, y_m=1.0) for p, x in (("a", 0), ("b", 2), ("c", 4))]
    assert "collinear" in compute_gdop_warning(phones)
    phones[1].y_m = 4.0
    assert compute_gdop_warning(phones) is None
    phones[2].connected = False
    assert "at least 3" in compute_gdop_warning(phones)


def _hello(ws, x, y):
    ws.send_json({"type": "phone_hello", "phone_id": "p1", "x_m": x, "y_m": y, "sample_rate": 48000})
    for _ in range(10):
        if ws.receive_json()["type"] == "phone_status":
            return


def test_server_owns_phone_position_after_first_hello_or_put():
    session = Session()
    with TestClient(create_app(lambda: session)) as client:
        with client.websocket_connect("/ws/phone/p1") as ws:
            _hello(ws, 1.0, 1.0)
        assert (session.phones["p1"].x_m, session.phones["p1"].y_m) == (1.0, 1.0)
        assert client.put("/api/phones/p1", json={"x_m": 2.5, "y_m": 3.5}).status_code == 200
        with client.websocket_connect("/ws/phone/p1") as ws:
            _hello(ws, 1.0, 1.0)  # stale position from the phone's localStorage
        assert (session.phones["p1"].x_m, session.phones["p1"].y_m) == (2.5, 3.5)
