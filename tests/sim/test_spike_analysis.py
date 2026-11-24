"""Agent 3 phase 1: spike report on a recording with chirps then claps."""

from analysis.spike import analyze, format_report
from sim.room import DEFAULT_ROOM, SimEvent, default_phones, save_npz, simulate

import numpy as np


def test_spike_report_on_simulated_session(tmp_path):
    rng = np.random.default_rng(5)
    phones = default_phones(rng, 4, snr_db=25.0)
    phones[1].drop_ranges_s = [(20.0, 20.05)]
    room = DEFAULT_ROOM
    events = [SimEvent(t, room.speaker_x_m, room.speaker_y_m, "chirp") for t in (1.0, 11.0, 21.0)]
    claps = [(2.0, 2.0), (6.0, 4.0), (4.0, 3.0)]
    events += [SimEvent(24.0 + 2 * i, x, y, "clap") for i, (x, y) in enumerate(claps)]
    rec = simulate(room, phones, events, 31.0, reflection_gain=0.3, seed=5)
    path = save_npz(rec, tmp_path / "spike-sim.npz")

    report = analyze(path)
    assert set(report.phones) == {"p1", "p2", "p3", "p4"}
    assert report.phones["p2"].gap_count == 1
    assert {p.fs for p in report.phones.values()} == {44100, 48000}
    assert all(p.chirps == 3 for p in report.phones.values())
    assert max(p.residual_ms for p in report.phones.values()) < 0.1
    assert len(report.claps) == 3
    for clap in report.claps:
        assert clap.max_tdoa_error_ms < 0.3
    text = format_report(report)
    assert "Gate 0" in text and "p1" in text
