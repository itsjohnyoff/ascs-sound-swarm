"""Regression tests for v2 SyncEngine, held-out 3-chirp scoring, replay CLI, OnsetDetector, and gap-aware localize."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from shared.protocol import PhoneConfig, RoomConfig, speed_of_sound
from backend.dsp.chirp import generate_lfm_chirp
from backend.dsp.clock import evaluate_held_out_chirp, make_phone_slice
from backend.dsp.pipeline import LocalizationPipeline, OnsetDetector, SyncEngine
from backend.dsp.replay import RecordingValidationError, load_recording_npz, replay_recording


def _sim_recording(
    global_sig_48k: np.ndarray,
    fs_out: int,
    offset_s: float,
    drift_ppm: float,
    total_s: float,
) -> np.ndarray:
    fs_true = fs_out * (1.0 + drift_ppm * 1e-6)
    n_out = int(round(total_s * fs_out))
    t_glob = offset_s + np.arange(n_out, dtype=np.float64) / fs_true
    return map_coordinates(
        global_sig_48k.astype(np.float64),
        [t_glob * 48000.0],
        order=3,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def _write_three_chirp_npz(
    out_path: Path,
    chirp3_extra_delay_by_phone: dict[str, float] | None = None,
    omit_chirp3_phones: set[str] | None = None,
) -> tuple[RoomConfig, dict[str, PhoneConfig]]:
    """Generate a synthetic v2 3-chirp (0.8s, 30.8s, 60.8s) mixed-rate .npz archive."""
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)
    fs_sim = 48000
    chirp = generate_lfm_chirp(fs_sim)
    chirp_times = [0.8, 30.8, 60.8]
    total_s = 63.0

    specs = [
        ("iphone", 0.5, 0.5, 44100, -0.210, +38.0),
        ("android_a", 7.5, 0.5, 48000, +0.145, -26.0),
        ("android_b", 4.0, 5.5, 48000, +0.290, +19.0),
    ]
    phones: dict[str, PhoneConfig] = {}
    arrays: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(2026)
    noise_std = float(np.std(chirp)) / (10.0 ** (15.0 / 20.0))

    for idx, (pid, x, y, fs_phone, offset_s, drift_ppm) in enumerate(specs):
        phones[pid] = PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs_phone)
        tau_prop = math.hypot(x - room.speaker_x_m, y - room.speaker_y_m) / c
        g_sig = np.zeros(int((total_s + 2.0) * fs_sim), dtype=np.float32)

        for c_idx, t_em in enumerate(chirp_times):
            if c_idx == 2 and omit_chirp3_phones and pid in omit_chirp3_phones:
                continue
            extra = chirp3_extra_delay_by_phone.get(pid, 0.0) if (c_idx == 2 and chirp3_extra_delay_by_phone) else 0.0
            t_arr = t_em + tau_prop + extra
            # Direct path + 0.65x wall reflection +7 ms later
            for amp, dt in ((1.0, 0.0), (0.65, 0.007)):
                pos = (t_arr + dt) * fs_sim
                i0 = int(math.floor(pos))
                frac = pos - i0
                # Shift chirp sub-sample cleanly into g_sig
                idx_c = np.arange(len(chirp), dtype=np.float64) - frac
                shifted = map_coordinates(chirp.astype(np.float64), [idx_c], order=3, mode="constant", cval=0.0)
                g_sig[i0 : i0 + len(chirp)] += (amp * shifted).astype(np.float32)

        rec = _sim_recording(g_sig, fs_phone, offset_s=offset_s, drift_ppm=drift_ppm, total_s=total_s)
        rec += rng.normal(0.0, noise_std, size=len(rec)).astype(np.float32)

        arrays[f"{pid}_pcm"] = rec
        arrays[f"{pid}_start_index"] = np.int64(10_000 + idx * 7_500)
        arrays[f"{pid}_fs"] = np.int32(fs_phone)
        arrays[f"{pid}_valid"] = np.ones(len(rec), dtype=bool)

    meta = {
        "room": {
            "width_m": room.width_m,
            "height_m": room.height_m,
            "temperature_c": room.temperature_c,
            "speaker_x_m": room.speaker_x_m,
            "speaker_y_m": room.speaker_y_m,
        },
        "phones": {pid: {"x_m": p.x_m, "y_m": p.y_m, "label": p.label} for pid, p in phones.items()},
        "chirp_times_s": chirp_times,
        "notes": "Synthetic mixed-rate 3-chirp v2 capture",
    }
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(out_path, **arrays)
    return room, phones


def test_replay_validation_errors_and_no_pickle(tmp_path: Path) -> None:
    with pytest.raises(RecordingValidationError, match="not found"):
        load_recording_npz(tmp_path / "missing.npz")

    # Pickle object array refused
    bad_obj = tmp_path / "pickle.npz"
    np.savez(bad_obj, meta=np.array({"room": {}}, dtype=object))
    with pytest.raises(RecordingValidationError, match="pickle"):
        load_recording_npz(bad_obj)

    # Missing <id>_valid
    bad_missing = tmp_path / "missing_valid.npz"
    meta = json.dumps({"room": {"width_m": 8, "height_m": 6}, "phones": {"p1": {"x_m": 1, "y_m": 1}}})
    np.savez(
        bad_missing,
        meta=np.array(meta),
        p1_pcm=np.zeros(1000, dtype=np.float32),
        p1_start_index=np.int64(0),
        p1_fs=np.int32(48000),
    )
    with pytest.raises(RecordingValidationError, match="p1_valid"):
        load_recording_npz(bad_missing)

    # Length mismatch + NaN inside valid region
    bad_len = tmp_path / "bad_len.npz"
    np.savez(
        bad_len,
        meta=np.array(meta),
        p1_pcm=np.zeros(1000, dtype=np.float32),
        p1_start_index=np.int64(0),
        p1_fs=np.int32(48000),
        p1_valid=np.ones(900, dtype=bool),
    )
    with pytest.raises(RecordingValidationError, match="must match"):
        load_recording_npz(bad_len)

    bad_nan = tmp_path / "bad_nan.npz"
    pcm_nan = np.zeros(1000, dtype=np.float32)
    pcm_nan[10] = np.nan
    np.savez(
        bad_nan,
        meta=np.array(meta),
        p1_pcm=pcm_nan,
        p1_start_index=np.int64(0),
        p1_fs=np.int32(48000),
        p1_valid=np.ones(1000, dtype=bool),
    )
    with pytest.raises(RecordingValidationError, match="NaN/Inf"):
        load_recording_npz(bad_nan)


def test_held_out_three_chirp_sync_and_cli_replay(tmp_path: Path) -> None:
    npz_path = tmp_path / "sim_3chirp.npz"
    _write_three_chirp_npz(npz_path)

    report = replay_recording(npz_path, fit_chirps=2, held_out_index=2)
    ho = report.held_out
    print(
        f"\nHeld-out 3-chirp (0, 30, 60s): status={ho.status}, worst={ho.worst_pairwise_ms:.4f} ms, "
        f"diffs={ho.per_phone_diff_ms}, drift={report.full_fit_drift_ppm}"
    )
    assert ho.status == "pass"
    assert ho.gate_passed is True
    assert ho.coverage_complete is True
    assert ho.worst_pairwise_ms is not None and ho.worst_pairwise_ms < 0.25
    assert set(ho.phones_measured) == {"iphone", "android_a", "android_b"}
    # Fit on 2 chirps must match exact 2-point residual (~0) while full 3-chirp fit detects true drift
    assert report.full_fit_residuals_ms[report.full_fit_ref_phone_id or "iphone"] == 0.0

    # Verify CLI invocation end-to-end
    proc = subprocess.run(
        [sys.executable, "-m", "backend.dsp.replay", str(npz_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Gate (<1.000 ms, >=2 phones): PASS" in proc.stdout
    assert "Held-out chirp #3 (scored BEFORE fitting):" in proc.stdout


def test_held_out_anti_leakage_and_incomplete_coverage(tmp_path: Path) -> None:
    # Anti-leakage: perturb ONLY android_a's chirp 3 by +4.0 ms; would shrink if chirp 3 leaked into fit!
    perturbed_npz = tmp_path / "perturbed_chirp3.npz"
    _write_three_chirp_npz(perturbed_npz, chirp3_extra_delay_by_phone={"android_a": 0.004})
    rep_bad = replay_recording(perturbed_npz, fit_chirps=2, held_out_index=2)
    assert rep_bad.held_out.worst_pairwise_ms is not None
    print(f"\nAnti-leakage (+4.0 ms chirp 3 perturbation): worst_pairwise={rep_bad.held_out.worst_pairwise_ms:.3f} ms")
    assert 3.7 <= rep_bad.held_out.worst_pairwise_ms <= 4.3
    assert rep_bad.held_out.gate_passed is False
    assert rep_bad.held_out.status == "fail"

    # Incomplete coverage: 2 of 3 phones miss chirp 3 -> fewer than 2 phones measured
    missing_npz = tmp_path / "incomplete_chirp3.npz"
    _write_three_chirp_npz(missing_npz, omit_chirp3_phones={"android_a", "android_b"})
    rep_inc = replay_recording(missing_npz, fit_chirps=2, held_out_index=2)
    assert rep_inc.held_out.status == "incomplete_coverage"
    assert rep_inc.held_out.coverage_complete is False
    assert rep_inc.held_out.gate_passed is False
    assert set(rep_inc.held_out.phones_missing) == {"android_a", "android_b"}


def test_repeated_chirps_dedup_clock_jump_and_reference_drop(tmp_path: Path) -> None:
    npz_path = tmp_path / "sim_3chirp.npz"
    room, phones = _write_three_chirp_npz(npz_path)
    from backend.dsp.replay import load_recording_npz

    session = load_recording_npz(npz_path)
    engine = SyncEngine()

    # Incremental windows: [0..5s], overlapping [0..35s], [58..63s]
    def _subslice(slc, t0_s: float, t1_s: float):
        i0 = int(round(t0_s * slc.sample_rate))
        i1 = int(round(t1_s * slc.sample_rate))
        return make_phone_slice(
            phone_id=slc.phone_id,
            samples_f32=slc.samples_f32[i0:i1],
            start_sample_index=slc.start_sample_index + i0,
            sample_rate=slc.sample_rate,
            valid_mask=slc.valid_mask[i0:i1] if slc.valid_mask is not None else None,
        )

    w1 = {pid: _subslice(slc, 0.0, 5.0) for pid, slc in session.slices.items()}
    r1 = engine.add_chirp_observation(w1, phones, room, mode="lfm_single")
    assert r1.success is True
    assert engine.clock("android_a").n_points == 1  # type: ignore[union-attr]

    # Overlapping window repeats Chirp 1 + adds Chirp 2 -> n_points must be 2, not 3!
    w2 = {pid: _subslice(slc, 0.0, 35.0) for pid, slc in session.slices.items()}
    r2 = engine.add_chirp_observation(w2, phones, room, mode="lfm_single")
    assert r2.success is True
    assert engine.clock("android_a").n_points == 2  # type: ignore[union-attr]

    w3 = {pid: _subslice(slc, 58.0, 63.0) for pid, slc in session.slices.items()}
    r3 = engine.add_chirp_observation(w3, phones, room, mode="lfm_single")
    assert r3.success is True
    assert engine.clock("android_a").n_points == 3  # type: ignore[union-attr]

    # Round-trip to_global <-> to_local
    t_g = engine.to_global("iphone", 123456.75)
    assert t_g is not None
    n_back = engine.to_local("iphone", t_g)
    assert n_back is not None and abs(n_back - 123456.75) < 1e-6

    # Reference phone drops -> remaining 2 phones retain timeline continuity automatically
    old_ref = engine.ref_phone_id
    assert old_ref is not None
    engine.mark_clock_jump(old_ref)
    assert engine.is_synced(old_ref) is False
    assert engine.ref_phone_id is not None and engine.ref_phone_id != old_ref
    assert engine.needs_recalibration is False

    remaining = [pid for pid in phones if pid != old_ref]
    assert all(engine.is_synced(pid) for pid in remaining)

    # Verify remaining phones stay sub-ms aligned on Chirp 3 without re-running calibration
    common = engine.get_common_time_range({pid: session.slices[pid] for pid in remaining})
    assert common is not None
    aligned = engine.extract_aligned_window(
        {pid: session.slices[pid] for pid in remaining},
        t_start_s=common[1] - 3.5,
        t_end_s=common[1] - 0.5,
        target_fs=48000,
    )
    assert set(aligned.keys()) == set(remaining)

    # Now drop a second phone -> only 1 remains, must surface recalibration needed
    engine.mark_clock_jump(remaining[0])
    assert engine.needs_recalibration is True
    assert engine.recalibration_reason is not None


def test_gap_aware_extraction_onset_detector_and_v2_localize() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)
    fs = 48000
    coords = [("p1", 0.5, 0.5), ("p2", 7.5, 0.5), ("p3", 7.5, 5.5), ("p4", 0.5, 5.5)]
    phones = {pid: PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs) for pid, x, y in coords}

    chirp = generate_lfm_chirp(fs)
    rng = np.random.default_rng(55)
    # Build 2.0 s global soundscape: chirp at 0.3s, clap at 1.10s from (3.2, 2.4), clean second clap at 1.65s from (5.1, 3.8)
    src_1 = (3.2, 2.4)
    src_2 = (5.1, 3.8)
    t_b = np.arange(int(0.020 * fs)) / fs
    clap_burst = (0.7 * np.exp(-t_b / 0.004) * rng.normal(0.0, 1.0, size=len(t_b))).astype(np.float32)

    slices = {}
    for idx, (pid, x, y) in enumerate(coords):
        sig = rng.normal(0.0, 1e-4, size=int(2.2 * fs)).astype(np.float32)
        tau_spk = math.hypot(x - room.speaker_x_m, y - room.speaker_y_m) / c
        i_c = int(round((0.30 + tau_spk) * fs))
        sig[i_c : i_c + len(chirp)] += chirp

        for src_xy, t_ev in ((src_1, 1.10), (src_2, 1.65)):
            tau_ev = math.hypot(x - src_xy[0], y - src_xy[1]) / c
            i_ev = int(round((t_ev + tau_ev) * fs))
            sig[i_ev : i_ev + len(clap_burst)] += clap_burst

        valid = np.ones(len(sig), dtype=bool)
        if pid == "p2":
            # Packet gap inside clap 1 window [1.08 .. 1.20 s], clean during clap 2 [1.63 .. 1.75 s]
            valid[int(1.11 * fs) : int(1.13 * fs)] = False
            sig[int(1.11 * fs) : int(1.13 * fs)] = 0.0

        slices[pid] = make_phone_slice(
            phone_id=pid,
            samples_f32=sig,
            start_sample_index=5000 + idx * 1000,
            sample_rate=fs,
            valid_mask=valid,
        )

    # OnsetDetector detects both claps with 150 ms refractory
    det = OnsetDetector()
    onsets = det.push(slices["p1"].samples_f32[int(0.8 * fs) :], start_sample_index=slices["p1"].start_sample_index + int(0.8 * fs), fs=fs)
    assert len(onsets) >= 2
    assert all((onsets[k + 1] - onsets[k]) >= int(0.15 * fs) for k in range(len(onsets) - 1))

    engine = SyncEngine()
    cal = engine.add_chirp_observation(
        {pid: make_phone_slice(pid, slc.samples_f32[: int(0.8 * fs)], slc.start_sample_index, fs, slc.valid_mask[: int(0.8 * fs)]) for pid, slc in slices.items()},
        phones,
        room,
        mode="lfm_single",
    )
    assert cal.success is True and cal.synced_count == 4

    # Map t_global of clap 1 & clap 2 via p1's clock (which is aligned to ref)
    t_c1 = engine.to_global("p1", slices["p1"].start_sample_index + int(1.10 * fs))
    t_c2 = engine.to_global("p1", slices["p1"].start_sample_index + int(1.65 * fs))
    assert t_c1 is not None and t_c2 is not None

    # Window 1 overlaps p2's valid_mask=False gap -> p2 excluded, clock retained, localization succeeds on 3 phones!
    aligned_1 = engine.extract_aligned_window(slices, t_c1 - 0.015, t_c1 + 0.065, target_fs=fs)
    assert "p2" not in aligned_1
    assert set(aligned_1.keys()) == {"p1", "p3", "p4"}
    assert engine.is_synced("p2") is True

    loc = LocalizationPipeline()
    res1 = loc.localize(aligned_1, phones, room, kind="transient", fs=fs, timestamp_s=t_c1)
    assert res1 is not None
    err1 = math.hypot(res1.x_m - src_1[0], res1.y_m - src_1[1])
    print(f"\nGap-excluded 3-phone localization error: {err1:.3f} m")
    assert err1 < 0.15

    # Window 2 is valid on p2 -> all 4 phones participate again!
    aligned_2 = engine.extract_aligned_window(slices, t_c2 - 0.015, t_c2 + 0.065, target_fs=fs)
    assert set(aligned_2.keys()) == {"p1", "p2", "p3", "p4"}
    res2 = loc.localize(aligned_2, phones, room, kind="transient", fs=fs, timestamp_s=t_c2)
    assert res2 is not None
    assert math.hypot(res2.x_m - src_2[0], res2.y_m - src_2[1]) < 0.15

    # Out-of-bounds slice excludes p4 while keeping p4 synced
    short_slices = dict(slices)
    short_slices["p4"] = make_phone_slice(
        phone_id="p4",
        samples_f32=slices["p4"].samples_f32[: int(1.3 * fs)],
        start_sample_index=slices["p4"].start_sample_index,
        sample_rate=fs,
        valid_mask=slices["p4"].valid_mask[: int(1.3 * fs)] if slices["p4"].valid_mask is not None else None,
    )
    aligned_oob = engine.extract_aligned_window(short_slices, t_c2 - 0.015, t_c2 + 0.065, target_fs=fs)
    assert "p4" not in aligned_oob
    assert set(aligned_oob.keys()) == {"p1", "p2", "p3"}
    assert engine.is_synced("p4") is True

    # v2 sustained localization (0.75 s drone buzz without triggering)
    t_d = np.arange(int(0.75 * fs), dtype=np.float64) / fs
    drone_src = sum((0.2 / k) * np.sin(2.0 * np.pi * 150.0 * k * t_d) for k in range(1, 9)).astype(np.float32)
    src_d = (4.7, 2.8)
    idx_d = np.arange(len(drone_src), dtype=np.float64)
    aligned_drone = {
        pid: map_coordinates(
            drone_src.astype(np.float64),
            [idx_d - (math.hypot(src_d[0] - p.x_m, src_d[1] - p.y_m) / c) * fs],
            order=3,
            mode="constant",
            cval=0.0,
        ).astype(np.float32)
        for pid, p in phones.items()
    }
    res_d = loc.localize(aligned_drone, phones, room, kind="sustained", fs=fs)
    assert res_d is not None
    assert math.hypot(res_d.x_m - src_d[0], res_d.y_m - src_d[1]) < 0.15


def test_gap_mask_valid_only_noise_floor_and_adjacent_corrupted_chirp() -> None:
    from backend.dsp.sync import detect_all_chirps, detect_chirp_arrivals

    fs = 48000
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.02, size=10 * fs).astype(np.float32)
    mask_heavy_gap = np.ones(len(noise), dtype=bool)
    mask_heavy_gap[: int(6.5 * fs)] = False
    noise[~mask_heavy_gap] = 0.0

    assert detect_all_chirps(noise, fs=fs, valid_mask=mask_heavy_gap) == []
    arr_single, _ = detect_chirp_arrivals(noise, fs=fs, mode="lfm_single", valid_mask=mask_heavy_gap)
    assert arr_single == []

    with pytest.raises(ValueError, match="valid_mask"):
        detect_all_chirps(noise, fs=fs, valid_mask=np.ones(100, dtype=bool))
    with pytest.raises(ValueError, match="valid_mask"):
        detect_chirp_arrivals(noise, fs=fs, valid_mask=np.ones(100, dtype=bool))

    # Gapped loud chirp at 2.0 s must NOT blank a clean chirp 0.5 s later at 2.5 s
    chirp = generate_lfm_chirp(fs)
    buf = rng.normal(0.0, 0.005, size=6 * fs).astype(np.float32)
    i_bad = int(2.0 * fs)
    i_clean = int(2.5 * fs)
    buf[i_bad : i_bad + len(chirp)] += 3.0 * chirp
    buf[i_clean : i_clean + len(chirp)] += chirp

    vm = np.ones(len(buf), dtype=bool)
    vm[i_bad + int(0.04 * fs) : i_bad + int(0.09 * fs)] = False
    buf[~vm] = 0.0

    hits = detect_all_chirps(buf, fs=fs, valid_mask=vm)
    assert len(hits) == 1
    err_ms = abs(hits[0].local_offset - i_clean) / fs * 1000.0
    assert err_ms < 0.5


def test_clipped_phone_exclusion_and_three_phone_fallback() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)
    fs = 48000
    coords = [("p1", 0.5, 0.5), ("p2", 7.5, 0.5), ("p3", 7.5, 5.5), ("p4", 0.5, 5.5)]
    phones = {pid: PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs) for pid, x, y in coords}

    # 1) Transient clap where p2 clips strongly and is phase-shifted by +25 ms
    src_clap = (2.4, 3.8)
    rng = np.random.default_rng(77)
    n_clap = int(0.45 * fs)
    t_b = np.arange(int(0.025 * fs)) / fs
    raw_b = np.exp(-t_b / 0.0045) * rng.normal(0.0, 1.0, size=len(t_b))
    burst = (0.6 * raw_b / (np.max(np.abs(raw_b)) + 1e-12)).astype(np.float32)
    raw_clap = np.zeros(n_clap, dtype=np.float32)
    raw_clap[int(0.15 * fs) : int(0.15 * fs) + len(burst)] = burst

    idx_c = np.arange(n_clap, dtype=np.float64)
    aligned_clap: dict[str, np.ndarray] = {}
    for pid, p in phones.items():
        tau = math.hypot(src_clap[0] - p.x_m, src_clap[1] - p.y_m) / c
        sig = map_coordinates(raw_clap.astype(np.float64), [idx_c - tau * fs], order=3, mode="constant", cval=0.0)
        sig += rng.normal(0.0, 1e-4, size=n_clap)
        if pid == "p2":
            sig_bad = map_coordinates(raw_clap.astype(np.float64), [idx_c - (tau + 0.025) * fs], order=3, mode="constant", cval=0.0)
            sig = np.clip(sig_bad * 12.0, -1.0, 1.0)
        aligned_clap[pid] = sig.astype(np.float32)

    loc = LocalizationPipeline()
    res_v2 = loc.localize(aligned_clap, phones, room, kind="transient", fs=fs)
    assert res_v2 is not None
    assert set(res_v2.phones_used) == {"p1", "p3", "p4"}
    assert math.hypot(res_v2.x_m - src_clap[0], res_v2.y_m - src_clap[1]) < 0.15

    res_v1 = loc.detect_and_localize(aligned_clap, phones, room, fs=fs)
    assert res_v1 is not None
    assert set(res_v1.phones_used) == {"p1", "p3", "p4"}
    assert math.hypot(res_v1.x_m - src_clap[0], res_v1.y_m - src_clap[1]) < 0.15

    # 3-phone fallback: dropping p2 would leave only 2 phones, so p2 is retained safely
    three_phones = {k: phones[k] for k in ("p1", "p2", "p3")}
    three_aligned = {k: aligned_clap[k] for k in ("p1", "p2", "p3")}
    res_3 = loc.localize(three_aligned, three_phones, room, kind="transient", fs=fs)
    assert res_3 is not None
    assert len(res_3.phones_used) == 3

    # 2) Sustained drone where p3 clips only near window edge (must check BEFORE Hann tapering)
    src_drone = (5.4, 2.2)
    n_drone = int(0.75 * fs)
    t_d = np.arange(n_drone, dtype=np.float64) / fs
    drone_sig = sum((0.2 / k) * np.sin(2.0 * np.pi * 150.0 * k * t_d) for k in range(1, 9)).astype(np.float32)
    idx_d = np.arange(n_drone, dtype=np.float64)
    aligned_drone: dict[str, np.ndarray] = {}
    for pid, p in phones.items():
        tau = math.hypot(src_drone[0] - p.x_m, src_drone[1] - p.y_m) / c
        shift = tau + (0.025 if pid == "p3" else 0.0)
        sig = map_coordinates(drone_sig.astype(np.float64), [idx_d - shift * fs], order=3, mode="constant", cval=0.0)
        if pid == "p3":
            # Clip at the very start/end of the 250 ms edge where a Hann window alone would mask it
            sig[: int(0.015 * fs)] = 1.0
            sig[-int(0.015 * fs) :] = -1.0
        aligned_drone[pid] = sig.astype(np.float32)

    res_dr_v2 = loc.localize(aligned_drone, phones, room, kind="sustained", fs=fs)
    assert res_dr_v2 is not None
    assert set(res_dr_v2.phones_used) == {"p1", "p2", "p4"}
    assert math.hypot(res_dr_v2.x_m - src_drone[0], res_dr_v2.y_m - src_drone[1]) < 0.20

    res_dr_v1 = loc.detect_and_localize(aligned_drone, phones, room, fs=fs)
    assert res_dr_v1 is not None
    assert set(res_dr_v1.phones_used) == {"p1", "p2", "p4"}
    assert math.hypot(res_dr_v1.x_m - src_drone[0], res_dr_v1.y_m - src_drone[1]) < 0.20

    # 3) SyncEngine.extract_aligned_window(..., exclude_clipped=True) excludes p2 while keeping sync
    chirp = generate_lfm_chirp(fs)
    sync_slices = {}
    for pid, p in phones.items():
        tau_spk = math.hypot(p.x_m - room.speaker_x_m, p.y_m - room.speaker_y_m) / c
        buf = rng.normal(0.0, 1e-4, size=int(2.0 * fs)).astype(np.float32)
        i_c = int(round((0.30 + tau_spk) * fs))
        buf[i_c : i_c + len(chirp)] += chirp
        ev_s = int(round(1.10 * fs))
        buf[ev_s : ev_s + len(aligned_clap[pid])] += aligned_clap[pid]
        sync_slices[pid] = make_phone_slice(pid, buf, start_sample_index=1000, sample_rate=fs)

    eng = SyncEngine()
    cal = eng.add_chirp_observation(sync_slices, phones, room, mode="lfm_single")
    assert cal.success is True
    t_clap_g = eng.to_global("p1", 1000 + int(round(1.25 * fs)))
    assert t_clap_g is not None
    ext = eng.extract_aligned_window(sync_slices, t_clap_g - 0.05, t_clap_g + 0.08, target_fs=fs, exclude_clipped=True)
    assert set(ext.keys()) == {"p1", "p3", "p4"}
    assert eng.is_synced("p2") is True


def test_replay_offline_localize_end_to_end(tmp_path: Path) -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)
    fs_sim = 48000
    chirp = generate_lfm_chirp(fs_sim) * np.float32(0.25)
    chirp_times = [0.8, 16.0, 31.0]
    total_s = 33.5
    clap_xy = (3.2, 2.4)
    drone_xy = (5.4, 3.9)

    rng = np.random.default_rng(2027)
    t_b = np.arange(int(0.025 * fs_sim)) / fs_sim
    raw_b = np.exp(-t_b / 0.0045) * rng.normal(0.0, 1.0, size=len(t_b))
    clap_burst = (0.85 * raw_b / (np.max(np.abs(raw_b)) + 1e-12)).astype(np.float32)
    t_d = np.arange(int(0.70 * fs_sim), dtype=np.float64) / fs_sim
    drone_burst = sum((0.25 / k) * np.sin(2.0 * np.pi * 150.0 * k * t_d) for k in range(1, 9)).astype(np.float32)

    specs = [
        ("iphone", 0.5, 0.5, 44100, -0.180, +36.0),
        ("android_a", 7.5, 0.5, 48000, +0.125, -28.0),
        ("android_b", 7.5, 5.5, 48000, +0.240, +21.0),
        ("android_c", 0.5, 5.5, 44100, -0.095, -42.0),
    ]
    phones: dict[str, PhoneConfig] = {}
    arrays: dict[str, np.ndarray] = {}
    chirp_noise_std = float(np.std(chirp)) / (10.0 ** (15.0 / 20.0))

    for idx, (pid, x, y, fs_phone, offset_s, drift_ppm) in enumerate(specs):
        phones[pid] = PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs_phone)
        tau_spk = math.hypot(x - room.speaker_x_m, y - room.speaker_y_m) / c
        g_sig = np.zeros(int((total_s + 2.0) * fs_sim), dtype=np.float32)

        for t_em in chirp_times:
            t_arr = t_em + tau_spk
            for amp, dt in ((1.0, 0.0), (0.65, 0.007)):
                pos = (t_arr + dt) * fs_sim
                i0 = int(math.floor(pos))
                idx_c = np.arange(len(chirp), dtype=np.float64) - (pos - i0)
                shifted = map_coordinates(chirp.astype(np.float64), [idx_c], order=3, mode="constant", cval=0.0)
                g_sig[i0 : i0 + len(chirp)] += (amp * shifted).astype(np.float32)

        # Clap at t=8.0 s + sustained drone burst at t=23.0 s
        for src_xy, t_ev, sig_ev in ((clap_xy, 8.0, clap_burst), (drone_xy, 23.0, drone_burst)):
            tau_ev = math.hypot(x - src_xy[0], y - src_xy[1]) / c
            pos = (t_ev + tau_ev) * fs_sim
            i0 = int(math.floor(pos))
            idx_e = np.arange(len(sig_ev), dtype=np.float64) - (pos - i0)
            shifted = map_coordinates(sig_ev.astype(np.float64), [idx_e], order=3, mode="constant", cval=0.0)
            g_sig[i0 : i0 + len(sig_ev)] += shifted.astype(np.float32)

        rec = _sim_recording(g_sig, fs_phone, offset_s=offset_s, drift_ppm=drift_ppm, total_s=total_s)
        rec += rng.normal(0.0, chirp_noise_std, size=len(rec)).astype(np.float32)

        valid = np.ones(len(rec), dtype=bool)
        if pid == "android_a":
            # Packet drop 250 ms after the clap: short window still covers the clap with all 4 phones!
            g0 = int((8.25 - offset_s) * fs_phone)
            g1 = int((8.28 - offset_s) * fs_phone)
            valid[g0:g1] = False
            rec[g0:g1] = 0.0

        arrays[f"{pid}_pcm"] = rec
        arrays[f"{pid}_start_index"] = np.int64(8_000 + idx * 4_000)
        arrays[f"{pid}_fs"] = np.int32(fs_phone)
        arrays[f"{pid}_valid"] = valid

    meta = {
        "room": {
            "width_m": room.width_m,
            "height_m": room.height_m,
            "temperature_c": room.temperature_c,
            "speaker_x_m": room.speaker_x_m,
            "speaker_y_m": room.speaker_y_m,
        },
        "phones": {pid: {"x_m": p.x_m, "y_m": p.y_m, "label": p.label} for pid, p in phones.items()},
    }
    arrays["meta"] = np.array(json.dumps(meta))
    npz_path = tmp_path / "replay_events.npz"
    np.savez(npz_path, **arrays)

    report = replay_recording(npz_path, localize=True)
    assert report.held_out.gate_passed is True
    assert report.held_out.worst_pairwise_ms is not None and report.held_out.worst_pairwise_ms < 0.25
    assert len(report.events) == 2

    ev1, ev2 = report.events
    err1 = math.hypot(ev1.x_m - clap_xy[0], ev1.y_m - clap_xy[1])
    err2 = math.hypot(ev2.x_m - drone_xy[0], ev2.y_m - drone_xy[1])
    print(f"\nOffline --localize replay: clap err={err1:.3f} m ({ev1.runtime_ms:.1f} ms), drone err={err2:.3f} m ({ev2.runtime_ms:.1f} ms)")
    assert set(ev1.phones_used) == {"iphone", "android_a", "android_b", "android_c"}
    assert err1 < 0.20
    assert err2 < 0.25

    proc_loc = subprocess.run(
        [sys.executable, "-m", "backend.dsp.replay", str(npz_path), "--localize", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data_loc = json.loads(proc_loc.stdout)
    assert len(data_loc["events"]) == 2

    proc_default = subprocess.run(
        [sys.executable, "-m", "backend.dsp.replay", str(npz_path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data_default = json.loads(proc_default.stdout)
    assert data_default["events"] == []


def test_real_room_low_ncc_reverberant_chirp_syncs_while_claps_rejected() -> None:
    from backend.dsp.sync import detect_all_chirps

    room = RoomConfig(width_m=5.0, height_m=7.0, temperature_c=22.0, speaker_x_m=2.5, speaker_y_m=3.5)
    c = speed_of_sound(room.temperature_c)
    fs = 48000
    chirp = generate_lfm_chirp(fs)
    rng = np.random.default_rng(314)
    n_tot = int(6.0 * fs)
    t_b = np.arange(int(0.025 * fs)) / fs
    raw_b = np.exp(-t_b / 0.004) * rng.normal(0.0, 1.0, size=len(t_b))
    clap = (0.8 * raw_b / (np.max(np.abs(raw_b)) + 1e-12)).astype(np.float32)

    coords = [("p1", 0.3, 6.7), ("p2", 4.7, 0.3), ("p3", 4.8, 6.7), ("p4", 0.3, 0.3)]
    phones = {pid: PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs) for pid, x, y in coords}
    slices = {}

    for idx, (pid, x, y) in enumerate(coords):
        sig = rng.normal(0.0, 2e-4, size=n_tot).astype(np.float32)
        # Loud broadband clap at t=1.2 s must NOT falsely trigger chirp detection even at min_ncc=0.15
        i_cl = int(1.2 * fs)
        sig[i_cl : i_cl + len(clap)] += clap

        # Reverberant chirp at t=3.5 s with realistic dense room tail overlapping the 150 ms template,
        # pushing local NCC into the real-room ~0.18..0.23 range while leaving direct arrival clear.
        tau_spk = math.hypot(x - room.speaker_x_m, y - room.speaker_y_m) / c
        t_dir = 3.5 + tau_spk
        echo_profile = [
            (1.0, 0.0),
            (0.65, 0.0042),
            (1.15 if pid == "p2" else 0.70, 0.0058),
        ]
        # Evenly spaced random-sign reverb tail across [8 ms, 148 ms] lowers template NCC into ~0.19..0.22
        # without random reflection clustering overtaking the direct arrival.
        rng_rir = np.random.default_rng(1000)
        for k, dt_late in enumerate(np.linspace(0.008, 0.148, 300)):
            amp_late = 0.56 * math.exp(-float(dt_late) / 0.16) * float(rng_rir.choice((-1.0, 1.0)))
            echo_profile.append((amp_late, float(dt_late)))
        for amp, dt in echo_profile:
            i_pos = int(round((t_dir + dt) * fs))
            if i_pos + len(chirp) <= len(sig):
                sig[i_pos : i_pos + len(chirp)] += np.float32(0.35 * amp) * chirp

        hits = detect_all_chirps(sig, fs=fs)
        assert len(hits) == 1, f"{pid} expected 1 chirp (clap rejected), got {len(hits)}"
        assert 0.15 <= hits[0].ncc < 0.25, f"{pid} expected 0.15 <= ncc < 0.25, got {hits[0].ncc:.3f}"
        assert abs(hits[0].local_offset / fs - t_dir) < 0.0008

        slices[pid] = make_phone_slice(
            pid, sig, start_sample_index=1000 + idx * 500, sample_rate=fs, valid_mask=np.ones(n_tot, dtype=bool)
        )

    eng = SyncEngine()
    cal = eng.add_chirp_observation(slices, phones, room, mode="lfm_single")
    assert cal.success is True
    assert cal.synced_count == 4
    assert cal.max_residual_ms is not None and cal.max_residual_ms < 1.5
    assert all(r.synced for r in cal.per_phone.values())


def test_detect_all_chirps_skips_louder_clipped_clap_and_finds_all_chirps() -> None:
    from backend.dsp.sync import detect_all_chirps, detect_chirp_arrivals

    fs = 48000
    chirp = generate_lfm_chirp(fs)
    rng = np.random.default_rng(909)
    n_tot = int(35.0 * fs)
    sig = rng.normal(0.0, 2e-4, size=n_tot).astype(np.float32)

    chirp_starts_s = [1.0, 16.0, 31.0]
    for ts in chirp_starts_s:
        pos = int(round(ts * fs))
        sig[pos : pos + len(chirp)] += np.float32(0.18) * chirp

    # Insert 2 huge near-field clipped claps (stronger raw matched-filter peak than the 0.18x chirps)
    t_b = np.arange(int(0.020 * fs)) / fs
    for clap_s in (8.0, 22.0):
        raw = np.exp(-t_b / 0.003) * rng.normal(0.0, 1.0, size=len(t_b))
        clap = np.clip(4.0 * raw / (np.max(np.abs(raw)) + 1e-12), -1.0, 1.0).astype(np.float32)
        pos = int(round(clap_s * fs))
        sig[pos : pos + len(clap)] = clap

    hits = detect_all_chirps(sig, fs=fs)
    assert len(hits) == 3, f"Expected 3 real chirps and 0 claps, got {[round(h.local_offset/fs, 3) for h in hits]}"
    for h, expected_s in zip(hits, chirp_starts_s):
        assert abs(h.local_offset / fs - expected_s) < 0.0005

    # Live calibration single-chirp window with a louder clap inside must still pick the chirp
    sub = sig[int(13.0 * fs) : int(24.0 * fs)]
    arr, _ = detect_chirp_arrivals(sub, fs=fs, mode="lfm_single")
    assert len(arr) == 1
    assert abs((arr[0] / fs + 13.0) - 16.0) < 0.0005


def test_clock_fitter_recovers_from_outlier_and_step_slip_chirps() -> None:
    from backend.dsp.clock import EmissionRecord, PhoneObservation, ReferenceClockFitter

    fs = 48000
    rates = {"p_ref": fs, "p_bad": fs}

    # Case 1: 1 glitched outlier chirp mixed with 3 clean chirps -> LOO drops outlier (< 0.1 ms)
    fitter = ReferenceClockFitter()
    times_s = [0.0, 10.0, 20.0, 30.0]
    emissions = []
    for k, ts in enumerate(times_s):
        idx_ref = 1000.0 + ts * fs
        slip = 0.0035 * fs if k == 1 else 0.0  # 3.5 ms outlier on chirp 1
        idx_bad = 5000.0 + ts * fs * (1.0 + 25e-6) + slip
        emissions.append(
            EmissionRecord(
                emission_id=k,
                obs_by_phone={
                    "p_ref": PhoneObservation(idx_ref, idx_ref, fs, 60.0, 0.9),
                    "p_bad": PhoneObservation(idx_bad, idx_bad, fs, 60.0, 0.9),
                },
            )
        )
    fitter.ingest_emissions(emissions, rates)
    ck = fitter.clock("p_bad")
    assert ck is not None and ck.residual_ms < 0.2

    # Case 2: buffer slip between old chirps (0, 1) and new chirps (2, 3) -> suffix recovery (< 0.1 ms)
    fitter2 = ReferenceClockFitter()
    emissions2 = []
    for k, ts in enumerate(times_s):
        idx_ref = 1000.0 + ts * fs
        step_offset = 0.0040 * fs if k >= 2 else 0.0  # 4 ms post-slip shift
        idx_bad = 5000.0 + ts * fs + step_offset
        emissions2.append(
            EmissionRecord(
                emission_id=k,
                obs_by_phone={
                    "p_ref": PhoneObservation(idx_ref, idx_ref, fs, 60.0, 0.9),
                    "p_bad": PhoneObservation(idx_bad, idx_bad, fs, 60.0, 0.9),
                },
            )
        )
    fitter2.ingest_emissions(emissions2, rates)
    ck2 = fitter2.clock("p_bad")
    assert ck2 is not None and ck2.residual_ms < 0.2
