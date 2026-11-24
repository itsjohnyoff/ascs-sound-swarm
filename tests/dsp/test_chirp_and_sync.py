"""TDD tests for LFM chirp synthesis, ggwave hybrid/benchmark, and sub-ms SyncEngine."""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy.signal import resample_poly

from shared.protocol import (
    CHIRP_AMPLITUDE,
    CHIRP_DURATION_S,
    CHIRP_F0_HZ,
    CHIRP_F1_HZ,
    CHIRP_TUKEY_ALPHA,
    DUAL_CHIRP_GAP_S,
    PhoneAudioSlice,
    PhoneConfig,
    RoomConfig,
    speed_of_sound,
)
from backend.dsp.chirp import (
    GGWAVE_AVAILABLE,
    benchmark_ggwave_vs_lfm,
    decode_ggwave_preamble,
    encode_ggwave_preamble,
    generate_dual_chirp,
    generate_hybrid_chirp,
    generate_lfm_chirp,
)
from backend.dsp.sync import SyncEngine, detect_chirp_arrivals


def _fractional_shift_and_resample(
    signal_48k: np.ndarray,
    fs_out: int,
    delay_s: float,
    drift_ppm: float,
    total_out_s: float,
) -> np.ndarray:
    """Simulate a phone recording a continuous global acoustic signal with offset and clock drift.

    Phone sample k occurs at global time:
        t_global(k) = delay_s + k / (fs_out * (1 + drift_ppm * 1e-6))
    where signal_48k[n] lives at t_global = n / 48000.
    """
    fs_true = fs_out * (1.0 + drift_ppm * 1e-6)
    n_out = int(round(total_out_s * fs_out))
    t_global = delay_s + np.arange(n_out, dtype=np.float64) / fs_true
    src_idx = t_global * 48000.0
    # Cubic interpolation via scipy.ndimage for clean ground-truth fractional delay
    from scipy.ndimage import map_coordinates

    return map_coordinates(
        signal_48k.astype(np.float64),
        [src_idx],
        order=3,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def test_lfm_chirp_contract_exactness() -> None:
    fs = 48000
    chirp = generate_lfm_chirp(fs)
    expected_n = int(round(fs * CHIRP_DURATION_S))
    assert chirp.dtype == np.float32
    assert chirp.shape == (expected_n,)
    assert abs(float(chirp[0])) < 1e-7
    assert abs(float(chirp[-1])) < 1e-7
    assert np.isclose(np.max(np.abs(chirp)), CHIRP_AMPLITUDE, atol=1e-3)

    # Verify exact midpoint phase matches contract formula
    k = expected_n // 2
    t = k / fs
    phase = 2.0 * math.pi * (
        CHIRP_F0_HZ * t
        + 0.5 * ((CHIRP_F1_HZ - CHIRP_F0_HZ) / CHIRP_DURATION_S) * t * t
    )
    assert np.isclose(float(chirp[k]), CHIRP_AMPLITUDE * math.sin(phase), atol=1e-5)


def test_dual_chirp_and_ggwave_wrappers() -> None:
    fs = 48000
    dual = generate_dual_chirp(fs)
    single = generate_lfm_chirp(fs)
    gap_samples = int(round(fs * DUAL_CHIRP_GAP_S))
    assert dual.shape == (gap_samples + len(single),)
    assert np.allclose(dual[: len(single)], single)
    assert np.allclose(dual[len(single) : gap_samples], 0.0)
    assert np.allclose(dual[gap_samples:], single)

    hybrid = generate_hybrid_chirp(fs)
    assert hybrid.ndim == 1 and len(hybrid) >= len(dual)

    if not GGWAVE_AVAILABLE:
        assert encode_ggwave_preamble("C01", fs) is None
        assert decode_ggwave_preamble(dual, fs) is None


def test_first_arrival_picks_direct_path_even_under_stronger_echo() -> None:
    fs = 48000
    rng = np.random.default_rng(42)
    chirp = generate_lfm_chirp(fs)
    true_start = 12345.37
    buf = np.zeros(int(1.2 * fs), dtype=np.float32)

    # Direct arrival + 1.05x stronger early reflection +7 ms later
    from scipy.ndimage import shift

    idx = np.arange(len(buf), dtype=np.float64)
    for amp, offset in ((1.0, true_start), (1.05, true_start + 0.007 * fs)):
        src_pos = idx - offset
        from scipy.ndimage import map_coordinates

        buf += amp * map_coordinates(chirp.astype(np.float64), [src_pos], order=3, mode="constant", cval=0.0).astype(
            np.float32
        )

    noise_std = float(np.std(chirp)) / (10.0 ** (15.0 / 20.0))
    buf += rng.normal(0.0, noise_std, size=len(buf)).astype(np.float32)

    arrivals, snr_db = detect_chirp_arrivals(buf, fs, mode="lfm_single")
    assert len(arrivals) == 1
    assert snr_db > 15.0
    err_ms = abs(arrivals[0] - true_start) / fs * 1000.0
    print(f"\nStronger-echo (+7ms, 1.05x) first-arrival error: {err_ms:.4f} ms (SNR={snr_db:.1f} dB)")
    assert err_ms < 0.15


def test_benchmark_ggwave_vs_lfm_runs_cleanly() -> None:
    fs = 48000
    chirp = generate_lfm_chirp(fs)
    clips = []
    delays = [0.050, 0.1234, 0.2107]
    for d in delays:
        buf = np.zeros(int(0.6 * fs), dtype=np.float32)
        s = int(round(d * fs))
        buf[s : s + len(chirp)] = chirp
        clips.append(buf)
    report = benchmark_ggwave_vs_lfm(clips, delays, fs)
    assert report["lfm_mae_ms"] < 0.05
    assert "ggwave_available" in report


def test_dual_chirp_sync_mixed_rates_offsets_drift_and_60s_holdover() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)

    specs = [
        ("p1", 0.5, 0.5, 48000, -0.240, +45.0),
        ("p2", 7.5, 0.5, 44100, +0.115, -38.0),
        ("p3", 7.5, 5.5, 48000, +0.380, +22.0),
        ("p4", 0.5, 5.5, 44100, -0.090, -55.0),
    ]
    phones = {
        pid: PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs)
        for pid, x, y, fs, _, _ in specs
    }

    # Build a 7.0 s global calibration soundscape at 48 kHz (chirps emitted at t_emit = 1.0 s and 6.0 s)
    fs_sim = 48000
    dual = generate_dual_chirp(fs_sim)
    t_emit_1 = 1.0
    rng = np.random.default_rng(7)

    cal_buffers: dict[str, PhoneAudioSlice] = {}
    start_indices: dict[str, int] = {"p1": 100_000, "p2": 50_000, "p3": 200_000, "p4": 10_000}

    for pid, x, y, fs_phone, offset_s, drift_ppm in specs:
        tau_prop = math.hypot(x - room.speaker_x_m, y - room.speaker_y_m) / c
        # Acoustic signal arriving at phone in global time: direct + 0.65x wall echo at +7 ms
        global_sig = np.zeros(int(8.5 * fs_sim), dtype=np.float32)
        i_dir = int(round((t_emit_1 + tau_prop) * fs_sim))
        i_echo = i_dir + int(round(0.007 * fs_sim))
        global_sig[i_dir : i_dir + len(dual)] += dual
        global_sig[i_echo : i_echo + len(dual)] += 0.65 * dual

        # Phone starts recording at global time = offset_s (plus random sample index base)
        rec = _fractional_shift_and_resample(global_sig, fs_phone, delay_s=offset_s, drift_ppm=drift_ppm, total_out_s=7.8)
        noise_std = float(np.std(dual[dual != 0])) / (10.0 ** (15.0 / 20.0))
        rec += rng.normal(0.0, noise_std, size=len(rec)).astype(np.float32)

        cal_buffers[pid] = PhoneAudioSlice(
            phone_id=pid,
            samples_f32=rec,
            start_sample_index=start_indices[pid],
            sample_rate=fs_phone,
        )

    engine = SyncEngine()
    res = engine.calibrate_from_buffers(cal_buffers, phones, room, mode="lfm_dual")
    assert res.success is True
    assert res.synced_count == 4
    assert res.median_residual_ms is not None and res.median_residual_ms < 0.25

    for pid, _, _, _, _, true_ppm in specs:
        rep = res.per_phone[pid]
        assert rep.synced is True
        assert rep.drift_ppm is not None
        assert abs(rep.drift_ppm - true_ppm) < 5.0, f"{pid} drift {rep.drift_ppm} vs {true_ppm}"

    # Now advance 60 seconds later and verify aligned extraction recovers simultaneous pulse within < 0.25 ms!
    t_pulse_global = 61.0
    # Create short 1.0 s slices around t = 60.5 .. 61.5 s using the EXACT same physical clock parameters
    from scipy.ndimage import map_coordinates

    # Build a broadband impulse-like transient at t = 61.0 s in global time
    t_local_win_start = 60.4
    later_buffers: dict[str, PhoneAudioSlice] = {}
    pulse_sig_48k = np.zeros(int(1.6 * fs_sim), dtype=np.float32)
    # 5 ms windowed tone burst at offset 0.6 s (= t_global 61.0 s)
    burst_t = np.arange(int(0.008 * fs_sim)) / fs_sim
    burst = (np.hanning(len(burst_t)) * np.sin(2.0 * np.pi * 2200.0 * burst_t)).astype(np.float32)
    pulse_sig_48k[int(0.6 * fs_sim) : int(0.6 * fs_sim) + len(burst)] = burst

    for pid, _, _, fs_phone, offset_s, drift_ppm in specs:
        fs_true = fs_phone * (1.0 + drift_ppm * 1e-6)
        k_start = int(round((t_local_win_start - offset_s) * fs_true))
        n_slice = int(round(1.2 * fs_phone))
        k_vec = k_start + np.arange(n_slice, dtype=np.float64)
        t_glob = offset_s + k_vec / fs_true
        src_idx = (t_glob - t_local_win_start) * fs_sim
        rec = map_coordinates(pulse_sig_48k.astype(np.float64), [src_idx], order=3, mode="constant", cval=0.0).astype(
            np.float32
        )
        later_buffers[pid] = PhoneAudioSlice(
            phone_id=pid,
            samples_f32=rec,
            start_sample_index=start_indices[pid] + k_start,
            sample_rate=fs_phone,
        )

    common = engine.get_common_time_range(later_buffers)
    assert common is not None
    t_lo, t_hi = common
    assert t_hi - t_lo > 0.8
    aligned = engine.extract_aligned_window(later_buffers, t_start_s=t_lo + 0.1, t_end_s=t_hi - 0.1, target_fs=48000)
    assert set(aligned.keys()) == {"p1", "p2", "p3", "p4"}

    # Peak cross-correlation lag between p1 and every other phone must be < 0.25 ms after 60 s!
    ref = aligned["p1"]
    max_err_ms = 0.0
    for pid in ("p2", "p3", "p4"):
        cc = np.correlate(aligned[pid], ref, mode="full")
        lag_samples = int(np.argmax(cc)) - (len(ref) - 1)
        err_ms = abs(lag_samples) / 48000.0 * 1000.0
        max_err_ms = max(max_err_ms, err_ms)
    print(f"\n60s holdover max inter-phone residual: {max_err_ms:.4f} ms (cal median={res.median_residual_ms:.4f} ms)")
    assert max_err_ms < 0.25

    # Clock jump invalidates phone immediately
    engine.mark_clock_jump("p2")
    assert engine.is_synced("p2") is False
    assert engine.is_synced("p1") is True


def _build_dual_chirp_phone_slices(
    phone_specs: list[tuple[str, float, float, int, float, float]],
    room: RoomConfig,
    t_emit_1: float = 1.0,
    total_out_s: float = 7.5,
    base_start_index: int = 10_000,
    seed: int = 11,
) -> tuple[dict[str, PhoneConfig], dict[str, PhoneAudioSlice]]:
    c = speed_of_sound(room.temperature_c)
    fs_sim = 48000
    dual = generate_dual_chirp(fs_sim)
    rng = np.random.default_rng(seed)
    phones: dict[str, PhoneConfig] = {}
    slices: dict[str, PhoneAudioSlice] = {}

    for idx, (pid, x, y, fs_phone, offset_s, drift_ppm) in enumerate(phone_specs):
        phones[pid] = PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs_phone)
        tau_prop = math.hypot(x - room.speaker_x_m, y - room.speaker_y_m) / c
        global_sig = np.zeros(int(math.ceil((t_emit_1 + total_out_s + 1.5) * fs_sim)), dtype=np.float32)
        i_dir = int(round((t_emit_1 + tau_prop) * fs_sim))
        global_sig[i_dir : i_dir + len(dual)] += dual

        rec = _fractional_shift_and_resample(
            global_sig, fs_phone, delay_s=offset_s, drift_ppm=drift_ppm, total_out_s=total_out_s
        )
        noise_std = float(np.std(dual[dual != 0])) / (10.0 ** (20.0 / 20.0))
        rec += rng.normal(0.0, noise_std, size=len(rec)).astype(np.float32)
        slices[pid] = PhoneAudioSlice(
            phone_id=pid,
            samples_f32=rec,
            start_sample_index=base_start_index + idx * 5_000,
            sample_rate=fs_phone,
        )
    return phones, slices


def test_dual_chirp_rejects_phone_missing_first_chirp_and_invalidates_prior_sync() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    specs = [
        ("phone_a", 0.5, 0.5, 48000, 0.0, 0.0),
        ("phone_b", 7.5, 0.5, 48000, 0.025, 0.0),
    ]
    phones, full_slices = _build_dual_chirp_phone_slices(specs, room, t_emit_1=1.0, total_out_s=7.5)

    # Cut phone_b between chirp 1 (t~1.0s) and chirp 2 (t~6.0s) so it hears only chirp 2
    cut_idx = int(round(3.0 * full_slices["phone_b"].sample_rate))
    short_b = PhoneAudioSlice(
        phone_id="phone_b",
        samples_f32=full_slices["phone_b"].samples_f32[cut_idx:].copy(),
        start_sample_index=full_slices["phone_b"].start_sample_index + cut_idx,
        sample_rate=full_slices["phone_b"].sample_rate,
    )
    bad_slices = {"phone_a": full_slices["phone_a"], "phone_b": short_b}

    # 1. Fresh dual-chirp calibration: phone_a synced, phone_b unsynced, overall success=False
    engine = SyncEngine()
    res = engine.calibrate_from_buffers(bad_slices, phones, room, mode="lfm_dual")
    assert res.per_phone["phone_a"].synced is True
    assert res.per_phone["phone_b"].synced is False
    assert "recalibrate" in (res.per_phone["phone_b"].error or "").lower()
    assert res.synced_count == 1
    assert res.success is False
    assert engine.is_synced("phone_a") is True
    assert engine.is_synced("phone_b") is False
    aligned = engine.extract_aligned_window(bad_slices, t_start_s=5.0, t_end_s=5.5, target_fs=48000)
    assert set(aligned.keys()) == {"phone_a"}

    # 2. Incomplete recalibration of a previously synced phone must invalidate its prior model
    recal_engine = SyncEngine()
    res_ok = recal_engine.calibrate_from_buffers(full_slices, phones, room, mode="lfm_dual")
    assert res_ok.success is True
    assert res_ok.synced_count == 2
    assert recal_engine.is_synced("phone_b") is True
    aligned_ok = recal_engine.extract_aligned_window(full_slices, t_start_s=0.5, t_end_s=1.0, target_fs=48000)
    assert set(aligned_ok.keys()) == {"phone_a", "phone_b"}

    # Advance sample indices for a second calibration round where phone_b again misses chirp 1
    _, round2_full = _build_dual_chirp_phone_slices(
        specs, room, t_emit_1=1.0, total_out_s=7.5, base_start_index=500_000, seed=19
    )
    cut2 = int(round(3.0 * round2_full["phone_b"].sample_rate))
    round2_bad = {
        "phone_a": round2_full["phone_a"],
        "phone_b": PhoneAudioSlice(
            phone_id="phone_b",
            samples_f32=round2_full["phone_b"].samples_f32[cut2:].copy(),
            start_sample_index=round2_full["phone_b"].start_sample_index + cut2,
            sample_rate=round2_full["phone_b"].sample_rate,
        ),
    }
    res_bad = recal_engine.calibrate_from_buffers(round2_bad, phones, room, mode="lfm_dual")
    assert res_bad.per_phone["phone_a"].synced is True
    assert res_bad.per_phone["phone_b"].synced is False
    assert "recalibrate" in (res_bad.per_phone["phone_b"].error or "").lower()
    assert res_bad.synced_count == 1
    assert res_bad.success is False
    assert recal_engine.is_synced("phone_a") is True
    assert recal_engine.is_synced("phone_b") is False

    # Previously synced phone_b must be excluded from aligned audio after incomplete recalibration
    common_a_start = recal_engine.sample_to_global_time("phone_a", round2_bad["phone_a"].start_sample_index)
    aligned_after = recal_engine.extract_aligned_window(
        round2_bad, t_start_s=common_a_start + 3.5, t_end_s=common_a_start + 4.0, target_fs=48000
    )
    assert set(aligned_after.keys()) == {"phone_a"}


def test_ggwave_hybrid_requires_both_chirps_and_other_phones_continue() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    specs = [
        ("phone_a", 0.5, 0.5, 48000, 0.0, 0.0),
        ("phone_b", 7.5, 0.5, 48000, 0.030, 0.0),
        ("phone_c", 4.0, 5.5, 44100, -0.050, +25.0),
    ]
    phones, slices = _build_dual_chirp_phone_slices(specs, room, t_emit_1=1.0, total_out_s=7.5, seed=23)
    cut_idx = int(round(3.0 * slices["phone_b"].sample_rate))
    slices["phone_b"] = PhoneAudioSlice(
        phone_id="phone_b",
        samples_f32=slices["phone_b"].samples_f32[cut_idx:].copy(),
        start_sample_index=slices["phone_b"].start_sample_index + cut_idx,
        sample_rate=slices["phone_b"].sample_rate,
    )

    engine = SyncEngine()
    res = engine.calibrate_from_buffers(slices, phones, room, mode="ggwave_hybrid")
    assert res.success is True
    assert res.synced_count == 2
    assert res.per_phone["phone_a"].synced is True
    assert res.per_phone["phone_c"].synced is True
    assert res.per_phone["phone_b"].synced is False
    assert "recalibrate" in (res.per_phone["phone_b"].error or "").lower()
    assert engine.is_synced("phone_b") is False

    common = engine.get_common_time_range(slices)
    assert common is not None
    aligned = engine.extract_aligned_window(slices, t_start_s=common[0] + 0.1, t_end_s=common[0] + 0.4, target_fs=48000)
    assert set(aligned.keys()) == {"phone_a", "phone_c"}


def test_lfm_single_mode_preserved() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)
    fs = 48000
    chirp = generate_lfm_chirp(fs)
    specs = [("phone_a", 0.5, 0.5, 0.020), ("phone_b", 7.5, 0.5, -0.040)]
    phones: dict[str, PhoneConfig] = {}
    slices: dict[str, PhoneAudioSlice] = {}

    for idx, (pid, x, y, offset_s) in enumerate(specs):
        phones[pid] = PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=fs)
        tau_prop = math.hypot(x - room.speaker_x_m, y - room.speaker_y_m) / c
        global_sig = np.zeros(int(2.0 * fs), dtype=np.float32)
        i_dir = int(round((0.5 + tau_prop) * fs))
        global_sig[i_dir : i_dir + len(chirp)] = chirp
        rec = _fractional_shift_and_resample(global_sig, fs, delay_s=offset_s, drift_ppm=0.0, total_out_s=1.5)
        slices[pid] = PhoneAudioSlice(
            phone_id=pid, samples_f32=rec, start_sample_index=10_000 + idx * 1000, sample_rate=fs
        )

    engine = SyncEngine()
    res = engine.calibrate_from_buffers(slices, phones, room, mode="lfm_single")
    assert res.success is True
    assert res.synced_count == 2
    assert res.per_phone["phone_a"].synced is True
    assert res.per_phone["phone_b"].synced is True
    assert res.max_residual_ms is not None and res.max_residual_ms < 0.25


def test_dual_chirp_rejects_full_length_noisy_phone_missing_chirp_1() -> None:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    c = speed_of_sound(room.temperature_c)
    fs = 48000
    n_samples = int(7.0 * fs)
    chirp = generate_lfm_chirp(fs)
    rng = np.random.default_rng(1234)

    phones = {
        "phone_a": PhoneConfig(phone_id="phone_a", label="phone_a", x_m=0.5, y_m=0.5, sample_rate=fs),
        "phone_b": PhoneConfig(phone_id="phone_b", label="phone_b", x_m=7.5, y_m=0.5, sample_rate=fs),
    }

    t_emit_1 = 0.8
    t_emit_2 = t_emit_1 + DUAL_CHIRP_GAP_S
    noise_std = float(np.std(chirp)) / (10.0 ** (20.0 / 20.0))

    slices: dict[str, PhoneAudioSlice] = {}
    for idx, (pid, include_chirp_1) in enumerate((("phone_a", True), ("phone_b", False))):
        pcfg = phones[pid]
        tau_prop = math.hypot(pcfg.x_m - room.speaker_x_m, pcfg.y_m - room.speaker_y_m) / c
        sig = rng.normal(0.0, noise_std, size=n_samples).astype(np.float32)
        if include_chirp_1:
            i1 = int(round((t_emit_1 + tau_prop) * fs))
            sig[i1 : i1 + len(chirp)] += chirp
        i2 = int(round((t_emit_2 + tau_prop) * fs))
        sig[i2 : i2 + len(chirp)] += chirp
        slices[pid] = PhoneAudioSlice(
            phone_id=pid,
            samples_f32=sig,
            start_sample_index=10_000 + idx * 5_000,
            sample_rate=fs,
        )

    arrivals_b, _ = detect_chirp_arrivals(slices["phone_b"].samples_f32, fs=fs, mode="lfm_dual")
    assert len(arrivals_b) == 1

    engine = SyncEngine()
    res = engine.calibrate_from_buffers(slices, phones, room, mode="lfm_dual")
    assert res.per_phone["phone_a"].synced is True
    assert res.per_phone["phone_b"].synced is False
    assert "recalibrate" in (res.per_phone["phone_b"].error or "").lower()
    assert res.synced_count == 1
    assert res.success is False
    assert engine.is_synced("phone_a") is True
    assert engine.is_synced("phone_b") is False
