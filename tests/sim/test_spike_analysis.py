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
    assert all(not p.agc_gated for p in report.phones.values())
    text = format_report(report)
    assert "Gate 0" in text and "p1" in text and "AGC/gate" in text


def test_spike_flags_noise_gated_phone(tmp_path):
    rng = np.random.default_rng(7)
    phones = default_phones(rng, 4, snr_db=45.0)
    room = DEFAULT_ROOM
    events = [SimEvent(t, room.speaker_x_m, room.speaker_y_m, "chirp") for t in (1.0, 5.0)]
    events.append(SimEvent(8.0, 4.0, 3.0, "clap"))
    rec = simulate(room, phones, events, 11.0, reflection_gain=0.2, seed=7)

    pcm = rec["p3_pcm"].copy()
    pcm[np.abs(pcm) < 0.005] = 0.0
    rec["p3_pcm"] = pcm
    path = save_npz(rec, tmp_path / "spike-gated.npz")

    report = analyze(path)
    assert report.phones["p3"].agc_gated is True
    assert report.phones["p1"].agc_gated is False


def test_spike_flags_agc_pumping_after_claps(tmp_path):
    rng = np.random.default_rng(11)
    phones = default_phones(rng, 4, snr_db=25.0)
    room = DEFAULT_ROOM
    events = [SimEvent(t, room.speaker_x_m, room.speaker_y_m, "chirp") for t in (1.0, 5.0)]
    clap_times = (8.0, 11.0, 14.0)
    events += [SimEvent(t, 4.0, 3.0, "clap") for t in clap_times]
    rec = simulate(room, phones, events, 17.0, reflection_gain=0.2, seed=11)

    pcm = rec["p4_pcm"].copy()
    fs = int(rec["p4_fs"])
    p4_start = phones[3].start_s
    for t_clap in clap_times:
        center = int(round((t_clap - p4_start) * fs))
        lo = max(0, center - int(0.2 * fs))
        hi = min(len(pcm), center + int(0.2 * fs))
        peak = lo + int(np.argmax(np.abs(pcm[lo:hi])))
        pcm[peak + int(0.15 * fs) : peak + int(0.45 * fs)] *= 0.25
    rec["p4_pcm"] = pcm
    path = save_npz(rec, tmp_path / "spike-agc.npz")

    report = analyze(path)
    assert report.phones["p4"].agc_gated is True
    assert report.phones["p1"].agc_gated is False
