from scripts.soak import MATCH_WINDOW_S, match_detections, soak_events
from shared.protocol import AUTO_CHIRP_PERIOD_S


def test_soak_events_chirp_every_period_with_two_targets_between():
    events = soak_events(600.0)
    chirps = [e.t_s for e in events if e.kind == "chirp"]
    assert chirps == [1.0 + k * AUTO_CHIRP_PERIOD_S for k in range(19)]
    targets = [e for e in events if e.kind != "chirp"]
    assert len(targets) == 2 * len(chirps)
    # Targets never land inside the chirp suppression zone.
    assert min(abs(t.t_s - c) for t in targets for c in chirps) >= 10.0
    assert {t.kind for t in targets} == {"clap", "drone"}


def test_match_detections_first_emission_latency_and_false_positives():
    targets = [e for e in soak_events(60.0) if e.kind != "chirp"]  # targets at 11 s and 21 s
    t0 = 100.0
    dets = [
        {"event_id": "a", "recv_s": t0 + 11.5, "x_m": targets[0].x_m + 0.3, "y_m": targets[0].y_m + 0.4},
        {"event_id": "a", "recv_s": t0 + 12.0, "x_m": 0.0, "y_m": 0.0},   # sustained re-emit: ignored
        {"event_id": "b", "recv_s": t0 + 21.0 + MATCH_WINDOW_S + 1, "x_m": 0.0, "y_m": 0.0},  # too late
    ]
    m = match_detections(dets, targets, t0, speed=1.0)
    assert m["targets"] == 2 and m["false"] == 1
    err, lat = m["hits"][0]
    assert abs(err - 0.5) < 1e-9 and abs(lat - 0.5) < 1e-9
    assert 1 not in m["hits"]
