"""Agent 3 phase 1: spike report on a recording with chirps then claps."""

import json

import numpy as np
import pytest

from analysis import spike
from analysis.spike import PhoneReport, SpikeReport, analyze, format_report
from sim.room import DEFAULT_ROOM, SimEvent, default_phones, save_npz, simulate


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
    assert "FAIL" in text and "p2: 1 sample-index gap" in text

    report = analyze(path, clap_points=claps[::-1])
    assert [clap.point for clap in report.claps] == claps[::-1]

    report = analyze(path, clap_points=claps[:2])
    assert all(clap.point is None for clap in report.claps)
    assert "Found 3 transient events for 2 taped claps" in format_report(report)


def clean_phone(**updates):
    return PhoneReport(**({"fs": 48000, "gap_count": 0, "chirps": 3, "residual_ms": 0.1,
                           "drift_ppm": 0.0, "clipped_fraction": 0.005, "fit_chirps": 3} | updates))


@pytest.mark.parametrize(("updates", "reason"), [
    ({"residual_ms": None}, "residual"),
    ({"residual_ms": 1.0}, "residual"),
    ({"chirps": 2}, "chirp"),
    ({"fit_chirps": 2}, "chirp"),
    ({"clipped_fraction": 0.01}, "clipped"),
    ({"gap_count": 1}, "sample-index gap"),
    ({"agc_gated": True, "agc_reason": "noise-gated"}, "noise-gated"),
    ({"dropped_frames": 1}, "dropped frame"),
    ({"segments": 2}, "segments"),
    ({"disconnects": 1}, "disconnect"),
])
def test_spike_report_rejects_failed_capture_condition(updates, reason):
    report = SpikeReport("spike-01.npz", "p1", {"p1": clean_phone(), "p2": clean_phone(**updates)}, [])

    assert any(reason in failure for failure in report.gate_failures)
    assert "FAIL" in format_report(report)


def test_spike_cli_accepts_sidecar_and_uses_gate_exit_status(tmp_path, monkeypatch, capsys):
    sidecar = tmp_path / "points.json"
    sidecar.write_text('[{"x_m": 2, "y_m": 3}]', encoding="utf-8")
    phones = {"p1": clean_phone(), "p2": clean_phone()}

    def analyze_with_sidecar(path, *, clap_points):
        assert clap_points == [(2.0, 3.0)]
        return SpikeReport(path, "p1", phones, [])

    monkeypatch.setattr(spike, "analyze", analyze_with_sidecar)
    monkeypatch.setattr(spike, "_plot", lambda *args: None)
    args = ["spike-01.npz", "--clap-points", str(sidecar), "--json"]
    assert spike.main(args) == 0
    assert json.loads(capsys.readouterr().out)["gate_failures"] == []

    del phones["p2"]
    assert spike.main(args) == 1
    assert "2–4 phones" in json.loads(capsys.readouterr().out)["gate_failures"][0]


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
