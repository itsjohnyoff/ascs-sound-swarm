"""Acoustic self-localization against simulated rooms with echoes, noise, drift and late emissions."""

import json

import numpy as np
import pytest

from backend.dsp.clock import ChirpArrival
from backend.dsp.ranging import LAPTOP, assign_slots, heard_from_audio, schedule, solve_layout
from shared.protocol import speed_of_sound
from sim.room import CENTRED_ROOM, locate_scenario

SPEAKER = (CENTRED_ROOM.speaker_x_m, CENTRED_ROOM.speaker_y_m)
C = speed_of_sound(CENTRED_ROOM.temperature_c)


def _heard(rec):
    meta = json.loads(str(rec["meta"]))
    ids = sorted(meta["phones"])
    n_slots = len(schedule(ids))
    heard = {p: heard_from_audio(rec[f"{p}_pcm"], int(rec[f"{p}_fs"]), meta["truth"]["phones"][p]["start_index"],
                                 n_slots) for p in ids}
    truth = {p: (meta["phones"][p]["x_m"], meta["phones"][p]["y_m"]) for p in ids}
    return ids, heard, truth


def _errors(result, truth):
    return [float(np.hypot(result.phones[p].x_m - x, result.phones[p].y_m - y)) for p, (x, y) in truth.items()]


def test_schedule_is_a_palindrome_framed_by_the_laptop():
    assert schedule(["a", "b", "c"]) == [LAPTOP, "a", "b", "c", "c", "b", "a", LAPTOP]


def test_assign_slots_tolerates_a_missing_chirp_and_late_starts():
    fs = 48000
    times = [10.0, 10.72, 11.38, 12.83, 13.55]  # slots 0, 1, 2, 4, 5: slot 3 never arrived
    arrivals = [ChirpArrival(sample_index=t * fs, local_offset=0, snr_db=30, ncc=0.9) for t in times]
    mapping = assign_slots(arrivals, fs, n_slots=6)
    assert sorted(mapping) == [0, 1, 2, 4, 5]
    assert mapping[4].sample_index == 12.83 * fs


@pytest.mark.parametrize("seed", range(4))
def test_four_phones_median_error_under_5_cm(seed):
    ids, heard, truth = _heard(locate_scenario(seed=seed))
    result = solve_layout(heard, ids, SPEAKER, C, prior=truth)
    assert result.success and result.anchored
    assert np.median(_errors(result, truth)) < 0.05
    assert all(f.status == "ok" for f in result.phones.values())


def test_three_phones_with_noise_and_strong_echoes():
    ids, heard, truth = _heard(locate_scenario(n_phones=3, snr_db=20.0, reflection_gain=0.5, seed=11))
    result = solve_layout(heard, ids, SPEAKER, C, prior=truth)
    assert result.success
    assert max(_errors(result, truth)) < 0.08


def test_a_missed_chirp_still_places_every_phone():
    ids, heard, truth = _heard(locate_scenario(seed=3))
    heard[ids[1]].pop(3)   # p2 missed p3's forward chirp
    heard[ids[2]].pop(2)   # p3 missed p2's forward chirp
    result = solve_layout(heard, ids, SPEAKER, C, prior=truth)
    assert result.success and np.median(_errors(result, truth)) < 0.05


def test_without_the_laptop_chirp_the_shape_is_right_but_unanchored():
    ids, heard, truth = _heard(locate_scenario(seed=4))
    last = len(schedule(ids)) - 1
    for p in ids:
        heard[p].pop(0, None)
        heard[p].pop(last, None)
    result = solve_layout(heard, ids, SPEAKER, C, prior=truth)
    assert result.success and not result.anchored
    for key, d in result.distances_m.items():
        a, b = key.split("|")
        assert abs(d - np.hypot(truth[a][0] - truth[b][0], truth[a][1] - truth[b][1])) < 0.05


def test_orientation_follows_the_prior_layout_including_a_mirror():
    ids, heard, truth = _heard(locate_scenario(seed=5))
    sx, sy = SPEAKER
    mirrored = {p: (2 * sx - x, y) for p, (x, y) in truth.items()}
    result = solve_layout(heard, ids, SPEAKER, C, prior=mirrored)
    assert np.median(_errors(result, mirrored)) < 0.05


def test_too_few_phones_is_a_clear_failure():
    result = solve_layout({"a": {0: 1.0}, "b": {0: 1.0}}, ["a", "b"], SPEAKER, C)
    assert not result.success and "at least 3" in result.error
