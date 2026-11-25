"""Agent 2 phase 0 MUST on the shared simulator (sync_scenario)."""

import json
from itertools import combinations

import numpy as np

from backend.dsp.chirp_detect import detect_chirps
from backend.dsp.clock import fit_clocks
from backend.dsp.replay import _parse_phone_positions, _parse_room
from shared.protocol import PhoneConfig
from sim.room import sync_scenario


def test_sync_gate_on_simulator_mixed_rates_drift_echo_noise():
    rec = sync_scenario(n_phones=4, n_chirps=3, chirp_spacing_s=30.0, snr_db=15.0, echo=(0.007, 0.65), seed=7)
    meta = json.loads(str(rec["meta"]))
    truth = meta["truth"]["phones"]
    assert {p["fs"] for p in truth.values()} == {44100, 48000}
    room = _parse_room(meta)
    phones = {pid: PhoneConfig(phone_id=pid, x_m=p["x_m"], y_m=p["y_m"], sample_rate=truth[pid]["fs"])
              for pid, p in _parse_phone_positions(meta).items()}
    arrivals, rates, ends = {}, {}, {}
    for pid in phones:
        fs, start = truth[pid]["fs"], truth[pid]["start_index"]
        arrivals[pid] = detect_chirps(rec[f"{pid}_pcm"], fs, start_sample_index=start, valid_mask=rec[f"{pid}_valid"])
        assert len(arrivals[pid]) == 3, pid
        rates[pid], ends[pid] = fs, start + len(rec[f"{pid}_pcm"])
    models, _, fitter = fit_clocks(arrivals, rates, phones, room, slice_end_indices=ends)
    assert set(models) == set(phones)

    residuals = {pid: m.residual_ms for pid, m in models.items()}
    print("fit residual ms:", {k: round(v, 4) for k, v in residuals.items()})
    assert max(residuals.values()) < 0.1

    t_eval = meta["chirp_times_s"][-1] + 120.0
    t_global = {}
    for pid, p in truth.items():
        n_local = p["start_index"] + (t_eval - p["start_s"]) * p["fs"] * (1 + p["drift_ppm"] * 1e-6)
        t_global[pid] = fitter.to_global(pid, n_local)
    worst_ms = max(abs(t_global[a] - t_global[b]) for a, b in combinations(truth, 2)) * 1e3
    print(f"alignment error 120 s after last chirp: {worst_ms:.4f} ms")
    assert worst_ms < 0.3
