"""Gate 0 offline synchronization tests: 3-chirp detection, reference clock fit, +120s holdover, and missed chirps."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from shared.protocol import PhoneConfig, RoomConfig, speed_of_sound
from backend.dsp.chirp import generate_lfm_chirp
from backend.dsp.chirp_detect import detect_chirps
from backend.dsp.clock import fit_clocks, flight_time_s, make_phone_slice
from backend.dsp.sync import SyncEngine


def _simulate_phone_audio(
    room: RoomConfig,
    phones: dict[str, PhoneConfig],
    specs: list[tuple[str, int, float, float, set[int]]],
    chirp_times_s: list[float],
    total_s: float = 64.0,
    echo_delay_s: float = 0.007,
    echo_gain: float = 0.65,
    snr_db: float = 15.0,
    seed: int = 2026,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    """Simulate full recordings for phones with mixed rates, offsets, drift, echo, noise, and optional missed chirps."""
    c = speed_of_sound(room.temperature_c)
    fs_sim = 48000
    chirp = generate_lfm_chirp(fs_sim)
    rng = np.random.default_rng(seed)
    noise_std = float(np.std(chirp)) / (10.0 ** (snr_db / 20.0))

    pcm_by_phone: dict[str, np.ndarray] = {}
    fs_by_phone: dict[str, int] = {}
    start_indices: dict[str, int] = {}

    for idx, (pid, fs_phone, offset_s, drift_ppm, missed) in enumerate(specs):
        pcfg = phones[pid]
        tau_prop = math.hypot(pcfg.x_m - room.speaker_x_m, pcfg.y_m - room.speaker_y_m) / c
        g_sig = np.zeros(int((total_s + 2.0) * fs_sim), dtype=np.float32)

        for c_idx, t_emit in enumerate(chirp_times_s):
            if c_idx in missed:
                continue
            t_arr = t_emit + tau_prop
            for amp, dt in ((1.0, 0.0), (echo_gain, echo_delay_s)):
                pos = (t_arr + dt) * fs_sim
                i0 = int(math.floor(pos))
                frac = pos - i0
                idx_c = np.arange(len(chirp), dtype=np.float64) - frac
                shifted = map_coordinates(chirp.astype(np.float64), [idx_c], order=3, mode="constant", cval=0.0)
                g_sig[i0 : i0 + len(chirp)] += (amp * shifted).astype(np.float32)

        fs_true = fs_phone * (1.0 + drift_ppm * 1e-6)
        n_out = int(round(total_s * fs_phone))
        t_glob = offset_s + np.arange(n_out, dtype=np.float64) / fs_true
        rec = map_coordinates(g_sig.astype(np.float64), [t_glob * fs_sim], order=3, mode="constant", cval=0.0).astype(
            np.float32
        )
        rec += rng.normal(0.0, noise_std, size=len(rec)).astype(np.float32)

        pcm_by_phone[pid] = rec
        fs_by_phone[pid] = fs_phone
        start_indices[pid] = 12_000 + idx * 8_000

    return pcm_by_phone, fs_by_phone, start_indices


def test_gate0_four_phones_mixed_rates_echo_120s_prediction() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=22.0, speaker_x_m=4.0, speaker_y_m=0.5)
    coords = [("p1", 0.5, 0.5), ("p2", 7.5, 0.5), ("p3", 7.5, 5.5), ("p4", 0.5, 5.5)]
    specs = [
        ("p1", 44100, -0.295, +50.0, set()),
        ("p2", 48000, +0.280, -48.0, set()),
        ("p3", 44100, -0.160, +35.0, set()),
        ("p4", 48000, +0.300, -50.0, set()),
    ]
    phones = {
        pid: PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs)
        for (pid, x, y), (_, fs, _, _, _) in zip(coords, specs)
    }
    chirp_times = [0.8, 30.8, 60.8]

    pcm_map, fs_map, start_map = _simulate_phone_audio(
        room=room,
        phones=phones,
        specs=specs,
        chirp_times_s=chirp_times,
        total_s=64.0,
        echo_delay_s=0.007,
        echo_gain=0.65,
        snr_db=15.0,
        seed=101,
    )

    # 1. Detect all 3 chirps on both 44.1 kHz and 48 kHz recordings, preferring direct path over +7 ms echo
    arrivals_by_phone = {}
    slice_ends = {}
    for pid, fs_phone, offset_s, drift_ppm, _ in specs:
        arrs = detect_chirps(pcm_map[pid], fs=fs_phone, start_sample_index=start_map[pid])
        assert len(arrs) == 3, f"{pid} expected 3 chirps, got {len(arrs)}"
        tau_prop = flight_time_s(phones[pid], room)
        fs_true = fs_phone * (1.0 + drift_ppm * 1e-6)
        for c_idx, arr in enumerate(arrs):
            expected_n = start_map[pid] + ((chirp_times[c_idx] + tau_prop - offset_s) * fs_true)
            err_ms = abs(arr.sample_index - expected_n) / fs_phone * 1000.0
            assert err_ms < 0.10, f"{pid} chirp {c_idx} direct-path error {err_ms:.4f} ms"
        arrivals_by_phone[pid] = arrs
        slice_ends[pid] = start_map[pid] + len(pcm_map[pid])

    # Verify optional explicit golden-vector template path produces identical detections
    tpl_44 = generate_lfm_chirp(44100)
    arrs_tpl = detect_chirps(pcm_map["p1"], fs=44100, template=tpl_44, start_sample_index=start_map["p1"])
    assert [round(a.sample_index, 4) for a in arrs_tpl] == [round(a.sample_index, 4) for a in arrivals_by_phone["p1"]]

    # 2. Fit clocks using reference phone (highest initial chirp SNR)
    models, emissions, fitter = fit_clocks(
        arrivals_by_phone=arrivals_by_phone,
        sample_rates=fs_map,
        phones=phones,
        room=room,
        slice_end_indices=slice_ends,
    )
    assert len(emissions) == 3
    expected_ref = max(arrivals_by_phone.keys(), key=lambda pid: arrivals_by_phone[pid][0].snr_db)
    assert fitter.ref_phone_id == expected_ref

    true_drift = {pid: d for pid, _, _, d, _ in specs}
    ref_true_drift = true_drift[expected_ref]
    print(f"\nGate 0 (4 phones, 44.1/48 kHz, +-300ms, +-50ppm, +7ms@0.65 echo, 15dB SNR), ref={expected_ref}:")
    for pid, model in sorted(models.items()):
        expected_rel_ppm = ((1.0 + ref_true_drift * 1e-6) / (1.0 + true_drift[pid] * 1e-6) - 1.0) * 1e6
        print(
            f"  {pid}: n_points={model.n_points}, residual={model.residual_ms:.4f} ms, "
            f"drift={model.drift_ppm:+.2f} ppm (true rel={expected_rel_ppm:+.2f} ppm)"
        )
        assert model.n_points == 3
        assert model.residual_ms < 0.1
        assert abs(model.drift_ppm - expected_rel_ppm) < 2.0

    # 3. Verify independent point 120 s after last chirp (t_emit = 60.8 + 120.0 = 180.8 s)
    c = speed_of_sound(room.temperature_c)
    t_indep = chirp_times[-1] + 120.0
    src_xy = (3.3, 4.1)
    pred_emit_s: dict[str, float] = {}
    for pid, fs_phone, offset_s, drift_ppm, _ in specs:
        tau_src = math.hypot(phones[pid].x_m - src_xy[0], phones[pid].y_m - src_xy[1]) / c
        fs_true = fs_phone * (1.0 + drift_ppm * 1e-6)
        n_true = start_map[pid] + (t_indep + tau_src - offset_s) * fs_true
        t_glob = fitter.to_global(pid, n_true)
        assert t_glob is not None
        pred_emit_s[pid] = t_glob - tau_src

    worst_120s_ms = (max(pred_emit_s.values()) - min(pred_emit_s.values())) * 1000.0
    print(f"  +120s independent point alignment spread: {worst_120s_ms:.4f} ms")
    assert worst_120s_ms < 0.3

    # Also verify cross-correlation alignment of an actual 180.8 s audio window via SyncEngine
    t_b_vec = np.arange(int(0.010 * 48000)) / 48000.0
    burst = (np.hanning(len(t_b_vec)) * np.sin(2.0 * np.pi * 2400.0 * t_b_vec)).astype(np.float32)
    burst_win = np.zeros(int(1.2 * 48000), dtype=np.float32)
    burst_win[int(0.6 * 48000) : int(0.6 * 48000) + len(burst)] = burst

    later_slices = {}
    t_win_start = t_indep - 0.6
    for pid, fs_phone, offset_s, drift_ppm, _ in specs:
        fs_true = fs_phone * (1.0 + drift_ppm * 1e-6)
        k0 = int(round((t_win_start - offset_s) * fs_true))
        n_slc = int(round(1.2 * fs_phone))
        k_vec = k0 + np.arange(n_slc, dtype=np.float64)
        t_g = offset_s + k_vec / fs_true
        rec = map_coordinates(burst_win.astype(np.float64), [(t_g - t_win_start) * 48000.0], order=3, mode="constant", cval=0.0).astype(np.float32)
        later_slices[pid] = make_phone_slice(pid, rec, start_map[pid] + k0, fs_phone)

    engine = SyncEngine()
    engine._v2_fitter = fitter
    common = engine.get_common_time_range(later_slices)
    assert common is not None
    aligned = engine.extract_aligned_window(later_slices, common[0] + 0.1, common[1] - 0.1, target_fs=48000)
    assert set(aligned.keys()) == {"p1", "p2", "p3", "p4"}
    ref_sig = aligned["p1"]
    max_cc_err_ms = 0.0
    for pid in ("p2", "p3", "p4"):
        cc = np.correlate(aligned[pid], ref_sig, mode="full")
        lag = int(np.argmax(cc)) - (len(ref_sig) - 1)
        max_cc_err_ms = max(max_cc_err_ms, abs(lag) / 48000.0 * 1000.0)
    print(f"  +120s waveform cross-correlation max lag: {max_cc_err_ms:.4f} ms")
    assert max_cc_err_ms < 0.3


def test_gate0_missed_chirps_do_not_shift_later_pairings() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    coords = [("p1", 0.5, 0.5), ("p2", 7.5, 0.5), ("p3", 7.5, 5.5), ("p4", 0.5, 5.5)]
    specs = [
        ("p1", 48000, 0.000, +10.0, set()),       # hears chirps 0, 1, 2
        ("p2", 44100, -0.220, +42.0, {1}),        # misses chirp 1 (middle) -> hears 0, 2
        ("p3", 48000, +0.250, -35.0, {0}),        # misses chirp 0 (first) -> hears 1, 2
        ("p4", 44100, -0.110, 0.0, {1, 2}),       # hears only chirp 0 (1 matched chirp)
    ]
    phones = {
        pid: PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs)
        for (pid, x, y), (_, fs, _, _, _) in zip(coords, specs)
    }
    chirp_times = [0.8, 30.8, 60.8]

    pcm_map, fs_map, start_map = _simulate_phone_audio(
        room=room,
        phones=phones,
        specs=specs,
        chirp_times_s=chirp_times,
        total_s=64.0,
        echo_delay_s=0.007,
        echo_gain=0.65,
        snr_db=15.0,
        seed=303,
    )

    arrivals_by_phone = {
        pid: detect_chirps(pcm_map[pid], fs=fs_phone, start_sample_index=start_map[pid])
        for pid, fs_phone, _, _, _ in specs
    }
    assert len(arrivals_by_phone["p1"]) == 3
    assert len(arrivals_by_phone["p2"]) == 2
    assert len(arrivals_by_phone["p3"]) == 2
    assert len(arrivals_by_phone["p4"]) == 1

    # Call fit_clocks WITHOUT slice_end_indices to verify missed-chirp pairing via start_sample_index too
    models, emissions, fitter = fit_clocks(
        arrivals_by_phone=arrivals_by_phone,
        sample_rates=fs_map,
        phones=phones,
        room=room,
    )
    assert len(emissions) == 3
    assert set(emissions[0].obs_by_phone.keys()) == {"p1", "p2", "p4"}
    assert set(emissions[1].obs_by_phone.keys()) == {"p1", "p3"}
    assert set(emissions[2].obs_by_phone.keys()) == {"p1", "p2", "p3"}

    assert models["p2"].n_points == 2
    assert models["p3"].n_points == 2
    assert models["p4"].n_points == 1
    assert models["p4"].drift_ppm == 0.0
    assert models["p4"].beta == pytest.approx(fs_map[fitter.ref_phone_id] / fs_map["p4"], rel=1e-9)

    # Verify p1, p2, p3 agree within < 0.3 ms at +120 s despite missed chirps
    c = speed_of_sound(room.temperature_c)
    t_indep = chirp_times[-1] + 120.0
    src_xy = (4.2, 3.0)
    pred_s = {}
    for pid, fs_phone, offset_s, drift_ppm, _ in specs[:3]:
        tau_src = math.hypot(phones[pid].x_m - src_xy[0], phones[pid].y_m - src_xy[1]) / c
        fs_true = fs_phone * (1.0 + drift_ppm * 1e-6)
        n_true = start_map[pid] + (t_indep + tau_src - offset_s) * fs_true
        t_glob = fitter.to_global(pid, n_true)
        assert t_glob is not None
        pred_s[pid] = t_glob - tau_src

    spread_ms = (max(pred_s.values()) - min(pred_s.values())) * 1000.0
    print(f"\nMissed-chirp +120s alignment spread (p1, p2, p3): {spread_ms:.4f} ms")
    assert spread_ms < 0.3
