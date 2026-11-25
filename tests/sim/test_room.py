"""Simulator contract: CONTRACTS 8 format, exact geometry, ground truth in meta."""

import json

import numpy as np
from scipy.signal import fftconvolve

from backend.dsp.replay import load_recording_npz
from shared.chirp import generate_lfm_chirp
from shared.protocol import RoomConfig, speed_of_sound
from sim.room import SimEvent, SimPhone, judge_scenario, save_npz, simulate, sync_scenario

ROOM = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=22.0, speaker_x_m=4.0, speaker_y_m=0.5)


def _arrival(pcm, fs):
    t = generate_lfm_chirp(fs)
    corr = fftconvolve(pcm, t[::-1], mode="full")[len(t) - 1:]
    return int(np.argmax(corr))


def test_closer_phone_hears_earlier_by_delta_d_over_c():
    phones = [SimPhone("near", 1.0, 1.0), SimPhone("far", 7.0, 5.0)]
    rec = simulate(ROOM, phones, [SimEvent(0.5, 4.0, 1.0, "chirp")], duration_s=1.2, seed=1)
    fs = 48000
    d_near, d_far = np.hypot(3.0, 0.0), np.hypot(3.0, 4.0)
    expected = (d_far - d_near) / speed_of_sound(ROOM.temperature_c) * fs
    measured = _arrival(rec["far_pcm"], fs) - _arrival(rec["near_pcm"], fs)
    assert abs(measured - expected) <= 1.0


def test_meta_has_ground_truth_and_loads_with_replay(tmp_path):
    rec = sync_scenario(n_chirps=2, chirp_spacing_s=3.0, seed=2)
    meta = json.loads(str(rec["meta"]))
    truth = meta["truth"]
    assert set(truth["phones"]) == set(meta["phones"])
    for p in truth["phones"].values():
        assert {"start_s", "drift_ppm", "fs", "start_index"} <= set(p)
    assert [e["kind"] for e in truth["events"]] == ["chirp", "chirp"]
    path = tmp_path / "s.npz"
    save_npz(rec, path)
    session = load_recording_npz(path)
    assert set(session.phones) == set(truth["phones"])
    for pid, p in truth["phones"].items():
        assert session.slices[pid].sample_rate == p["fs"]
        assert session.slices[pid].start_sample_index == p["start_index"]


def test_dropped_range_is_zeroed_and_invalid():
    phones = [SimPhone("a", 1.0, 1.0, snr_db=20.0, drop_ranges_s=[(0.2, 0.3)])]
    rec = simulate(ROOM, phones, [], duration_s=0.5, seed=3)
    valid, pcm = rec["a_valid"], rec["a_pcm"]
    gap = slice(int(0.2 * 48000), int(0.3 * 48000))
    assert not valid[gap].any() and np.all(pcm[gap] == 0.0)
    assert valid[: int(0.19 * 48000)].all()


def test_judge_scenario_has_claps_drones_and_outside_points():
    rec = judge_scenario(seed=4)
    events = json.loads(str(rec["meta"]))["truth"]["events"]
    targets = [e for e in events if e["kind"] != "chirp"]
    assert len(targets) == 10
    assert sum(e["kind"] == "clap" for e in targets) == 5
    assert sum(e["kind"] == "drone" for e in targets) == 5
    assert sum(not e["inside_hull"] for e in targets) == 2
