"""CONTRACTS 4 event grouping and the offline event finder used by spike/benchmark."""

import json

import pytest

from backend.server.events import EventGrouper, find_events_offline
from backend.dsp.replay import _parse_phone_positions, _parse_room
from shared.protocol import PhoneAudioSlice, PhoneConfig
from sim.room import judge_scenario


def test_three_phones_within_spread_open_one_event_at_earliest_onset():
    g = EventGrouper(max_spread_s=0.03, min_phones=3)
    g.add("a", 10.010)
    g.add("b", 10.000)
    assert g.pop_ready(lambda pid: 10.1, slack_s=0.4) == []  # only 2 phones so far
    g.add("c", 10.020)
    g.add("d", 10.500)  # too late: not part of it
    (event,) = g.pop_ready(lambda pid: 99.0, slack_s=0.4)
    assert event.t0 == pytest.approx(10.0)
    assert set(event.onsets) == {"a", "b", "c"}


def test_event_waits_for_every_phone_to_pass_playout_slack():
    g = EventGrouper(max_spread_s=0.03, min_phones=3)
    for pid in "abc":
        g.add(pid, 5.0)
    buffered = {"a": 5.5, "b": 5.5, "c": 5.3}
    assert g.pop_ready(buffered.__getitem__, slack_s=0.4) == []
    buffered["c"] = 5.41
    assert len(g.pop_ready(buffered.__getitem__, slack_s=0.4)) == 1
    assert g.pop_ready(buffered.__getitem__, slack_s=0.4) == []


def test_two_phone_cluster_expires_without_event():
    g = EventGrouper(max_spread_s=0.03, min_phones=3)
    g.add("a", 1.0)
    g.add("b", 1.01)
    assert g.pop_ready(lambda pid: 50.0, slack_s=0.4) == []
    assert g.pending == []


def _load(rec):
    meta = json.loads(str(rec["meta"]))
    phones = {pid: PhoneConfig(phone_id=pid, x_m=p["x_m"], y_m=p["y_m"]) for pid, p in _parse_phone_positions(meta).items()}
    slices = {pid: PhoneAudioSlice(pid, rec[f"{pid}_pcm"], int(rec[f"{pid}_start_index"]), int(rec[f"{pid}_fs"]),
                                   rec[f"{pid}_valid"]) for pid in phones}
    return meta, phones, slices, _parse_room(meta)


def test_offline_finder_reports_each_judge_event_once_and_skips_chirps():
    rec = judge_scenario(seed=11)
    meta, phones, slices, room = _load(rec)
    detections = find_events_offline(slices, phones, room)
    targets = [e for e in meta["truth"]["events"] if e["kind"] != "chirp"]
    assert len(detections) == len(targets)
    kinds = [d.kind for d in detections]
    assert kinds == ["sustained" if e["kind"] == "drone" else "transient" for e in targets]


def test_onsets_inside_a_held_sustained_event_are_ignored():
    g = EventGrouper(max_spread_s=0.03, min_phones=3)
    g.hold_until = 12.0
    for pid in "abc":
        g.add(pid, 11.5)
    assert g.pending == []
    for pid in "abc":
        g.add(pid, 12.1)
    assert len(g.pending) == 1


def test_buffered_time_of_zero_counts_as_buffered():
    g = EventGrouper(max_spread_s=0.03, min_phones=3)
    for pid in "abc":
        g.add(pid, -0.5)
    assert len(g.pop_ready(lambda pid: 0.0, slack_s=0.4)) == 1
