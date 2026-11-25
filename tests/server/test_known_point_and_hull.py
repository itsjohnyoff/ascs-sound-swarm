"""CONTRACTS 6: known-point checks stored apart from trials; CSV gains inside_hull."""

import csv
import io

from fastapi.testclient import TestClient

from backend.server.app import create_app
from backend.server.judge_eval import inside_hull
from backend.server.session import Session
from shared.protocol import LocalizationResult


def _session():
    session = Session()
    for pid, x, y in (("a", 1, 1), ("b", 7, 1), ("c", 7, 5), ("d", 1, 5)):
        session.set_phone(pid, x_m=x, y_m=y, connected=True)
    session.record_detection(LocalizationResult(
        event_id="evt-1", timestamp_s=1.0, x_m=3.3, y_m=3.4, uncertainty_m=0.1, residual_ms=0.1,
        confidence=0.9, phones_used=["a", "b", "c", "d"], sound_class="clap"))
    return session


def test_inside_hull():
    square = [(1, 1), (7, 1), (7, 5), (1, 5)]
    assert inside_hull(3, 3, square) and inside_hull(1, 1, square)
    assert not inside_hull(0.4, 3, square)
    assert not inside_hull(3, 3, square[:2])


def test_known_point_is_separate_from_trials_and_csv_has_inside_hull():
    with TestClient(create_app(_session)) as client:
        body = {"label": "known", "gt_x_m": 3.0, "gt_y_m": 3.0, "expected_class": "clap", "event_id": "evt-1"}
        kp = client.post("/api/eval/known-point", json=body).json()
        assert abs(kp["error_m"] - 0.5) < 1e-9 and kp["inside_hull"] is True
        assert client.get("/api/eval/known-point").json()["checks"][0]["error_m"] == kp["error_m"]
        assert client.get("/api/eval/trials").json()["trials"] == []

        client.post("/api/eval/trials", json={**body, "gt_x_m": 0.4})
        rows = list(csv.DictReader(io.StringIO(client.get("/api/eval/export.csv").text)))
        assert rows[0]["inside_hull"] == "False"
