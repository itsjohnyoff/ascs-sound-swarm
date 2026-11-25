"""Tests for tunable DSP parameters and unchanged default behavior (Task 1)."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

from backend.dsp.chirp import generate_lfm_chirp
from backend.dsp.chirp_detect import detect_chirps
from backend.dsp.localizer import AcousticLocalizer2D
from backend.dsp.pipeline import LocalizationPipeline
from backend.dsp.sync import SyncEngine, detect_all_chirps
from shared.protocol import PhoneAudioSlice, PhoneConfig, RoomConfig, speed_of_sound


def _make_room_and_phones() -> tuple[RoomConfig, dict[str, PhoneConfig]]:
    room = RoomConfig(
        width_m=6.0,
        height_m=5.0,
        speaker_x_m=3.0,
        speaker_y_m=0.5,
        temperature_c=22.0,
    )
    phones = {
        "p1": PhoneConfig(phone_id="p1", label="P1", x_m=0.5, y_m=0.5, connected=True),
        "p2": PhoneConfig(phone_id="p2", label="P2", x_m=5.5, y_m=0.5, connected=True),
        "p3": PhoneConfig(phone_id="p3", label="P3", x_m=5.5, y_m=4.5, connected=True),
        "p4": PhoneConfig(phone_id="p4", label="P4", x_m=0.5, y_m=4.5, connected=True),
    }
    return room, phones


def _delay_add(buf: np.ndarray, sig: np.ndarray, start_s: float, fs: int, gain: float = 1.0) -> None:
    idx = int(round(start_s * fs))
    end = min(len(buf), idx + len(sig))
    if end > idx >= 0:
        buf[idx:end] += (gain * sig[: end - idx]).astype(np.float32)


def test_default_params_match_baseline_sync_and_localize() -> None:
    fs = 48000
    room, phones = _make_room_and_phones()
    c = speed_of_sound(room.temperature_c)

    rng = np.random.default_rng(42)
    chirp = generate_lfm_chirp(fs=fs)
    t_clap = np.arange(int(0.020 * fs), dtype=np.float64) / fs
    clap = (rng.standard_normal(len(t_clap)) * np.exp(-t_clap / 0.004)).astype(np.float32)
    clap /= float(np.max(np.abs(clap))) + 1e-12

    n_total = int(2.0 * fs)
    chirp_t0 = 0.25
    clap_t0 = 1.10
    clap_xy = (2.4, 3.1)

    buffers: dict[str, PhoneAudioSlice] = {}
    for pid, pcfg in phones.items():
        sig = (rng.standard_normal(n_total) * 1e-4).astype(np.float32)
        tau_spk = math.hypot(pcfg.x_m - room.speaker_x_m, pcfg.y_m - room.speaker_y_m) / c
        tau_clap = math.hypot(pcfg.x_m - clap_xy[0], pcfg.y_m - clap_xy[1]) / c
        _delay_add(sig, chirp, chirp_t0 + tau_spk, fs, gain=0.6)
        _delay_add(sig, clap, clap_t0 + tau_clap, fs, gain=0.8)
        buffers[pid] = PhoneAudioSlice(
            phone_id=pid,
            samples_f32=sig,
            sample_rate=fs,
            start_sample_index=0,
        )

    # Default vs explicit-default SyncEngine
    sync_default = SyncEngine()
    sync_explicit = SyncEngine(
        chirp_lookback_s=0.025,
        chirp_direct_path_ratio=0.55,
        chirp_f_low=1200.0,
        chirp_f_high=8500.0,
    )
    cal_default = sync_default.add_chirp_observation(buffers, phones, room, mode="lfm_single")
    cal_explicit = sync_explicit.add_chirp_observation(buffers, phones, room, mode="lfm_single")
    assert cal_default.success and cal_explicit.success

    # Exact arrival sample_index equality for detect_all_chirps and detect_chirps
    for pid in phones:
        a_def = detect_all_chirps(buffers[pid].samples_f32, fs=fs)
        a_exp = detect_all_chirps(
            buffers[pid].samples_f32,
            fs=fs,
            lookback_s=0.025,
            direct_path_ratio=0.55,
            f_low=1200.0,
            f_high=8500.0,
        )
        c_exp = detect_chirps(
            buffers[pid].samples_f32,
            fs=fs,
            lookback_s=0.025,
            direct_path_ratio=0.55,
            f_low=1200.0,
            f_high=8500.0,
        )
        assert len(a_def) == len(a_exp) == len(c_exp) >= 1
        assert [a.sample_index for a in a_def] == [a.sample_index for a in a_exp]
        assert [a.sample_index for a in a_def] == [a.sample_index for a in c_exp]

    # Default vs explicit-default LocalizationPipeline
    aligned = sync_default.extract_aligned_window(buffers, clap_t0 - 0.015, clap_t0 + 0.065, target_fs=fs)
    pipe_default = LocalizationPipeline()
    pipe_explicit = LocalizationPipeline(
        grid_spacing_m=0.15,
        f_min=300.0,
        f_max=6000.0,
        phat_beta=0.8,
        cycle_threshold_s=0.0015,
        loo_residual_threshold_s=0.0005,
        sustained_frame_s=0.250,
        sustained_max_frames=3,
        transient_pre_s=0.015,
        transient_post_s=0.065,
    )
    res_def = pipe_default.localize(aligned, phones, room, kind="transient", fs=fs, timestamp_s=clap_t0)
    res_exp = pipe_explicit.localize(aligned, phones, room, kind="transient", fs=fs, timestamp_s=clap_t0)
    assert res_def is not None and res_exp is not None
    assert (res_def.x_m, res_def.y_m, res_def.residual_ms) == (res_exp.x_m, res_exp.y_m, res_exp.residual_ms)


def test_chirp_direct_path_ratio_and_lookback_affect_echo_picking() -> None:
    fs = 48000
    chirp = generate_lfm_chirp(fs=fs)
    sig = np.zeros(int(1.0 * fs), dtype=np.float32)

    direct_t = 0.200
    echo_t = direct_t + 0.012  # 12 ms later, louder than direct path
    _delay_add(sig, chirp, direct_t, fs, gain=0.45)
    _delay_add(sig, chirp, echo_t, fs, gain=1.00)

    # Default direct_path_ratio=0.55 ignores the 0.45x direct path and picks the 1.0x echo
    arr_default = detect_all_chirps(sig, fs=fs, direct_path_ratio=0.55, lookback_s=0.025)
    assert len(arr_default) == 1
    assert abs(arr_default[0].sample_index / fs - echo_t) < 0.002

    # Lowering direct_path_ratio=0.40 with lookback_s=0.020 recovers the earlier direct path
    arr_direct = detect_all_chirps(sig, fs=fs, direct_path_ratio=0.40, lookback_s=0.020)
    assert len(arr_direct) == 1
    assert abs(arr_direct[0].sample_index / fs - direct_t) < 0.002

    # Short lookback (0.005 s < 12 ms echo delay) still picks the echo even at ratio 0.40
    arr_short_lb = detect_all_chirps(sig, fs=fs, direct_path_ratio=0.40, lookback_s=0.005)
    assert len(arr_short_lb) == 1
    assert abs(arr_short_lb[0].sample_index / fs - echo_t) < 0.002


def test_localizer_band_beta_and_cycle_threshold_are_wired() -> None:
    fs = 48000
    room, phones = _make_room_and_phones()
    c = speed_of_sound(room.temperature_c)

    pipe = LocalizationPipeline(
        f_min=500.0,
        f_max=4000.0,
        phat_beta=0.6,
        cycle_threshold_s=0.0005,
        loo_residual_threshold_s=0.0003,
        sustained_frame_s=0.200,
        sustained_max_frames=2,
        transient_pre_s=0.010,
        transient_post_s=0.050,
    )
    assert isinstance(pipe._localizer, AcousticLocalizer2D)
    assert pipe._localizer.f_min == 500.0
    assert pipe._localizer.f_max == 4000.0
    assert pipe._localizer.phat_beta == 0.6
    assert pipe._localizer.cycle_threshold_s == 0.0005
    assert pipe._localizer.loo_residual_threshold_s == 0.0003
    assert pipe.sustained_frame_s == 0.200
    assert pipe.sustained_max_frames == 2
    assert pipe._trigger.sustained_frame_s == 0.200
    assert pipe._trigger.sustained_max_frames == 2
    assert pipe._trigger.transient_pre_s == 0.010
    assert pipe._trigger.transient_post_s == 0.050

    # Create two band-separated sources: low-band (1000-2000 Hz) at src_lo, high-band (4500-5500 Hz) at src_hi
    rng = np.random.default_rng(7)
    n_win = int(0.120 * fs)
    src_lo = (1.5, 3.8)
    src_hi = (4.5, 1.2)
    raw_lo = rng.standard_normal(int(0.030 * fs))
    raw_hi = rng.standard_normal(int(0.030 * fs))
    sos_lo = butter(4, [1000.0, 2000.0], btype="bandpass", fs=fs, output="sos")
    sos_hi = butter(4, [4500.0, 5500.0], btype="bandpass", fs=fs, output="sos")
    burst_lo = sosfiltfilt(sos_lo, raw_lo).astype(np.float32)
    burst_lo /= float(np.max(np.abs(burst_lo))) + 1e-12
    burst_hi = sosfiltfilt(sos_hi, raw_hi).astype(np.float32)
    burst_hi /= float(np.max(np.abs(burst_hi))) + 1e-12

    aligned: dict[str, np.ndarray] = {}
    for pid, pcfg in phones.items():
        buf = (rng.standard_normal(n_win) * 1e-3).astype(np.float32)
        tau_lo = math.hypot(pcfg.x_m - src_lo[0], pcfg.y_m - src_lo[1]) / c
        tau_hi = math.hypot(pcfg.x_m - src_hi[0], pcfg.y_m - src_hi[1]) / c
        _delay_add(buf, burst_lo, 0.020 + tau_lo, fs, gain=0.8)
        _delay_add(buf, burst_hi, 0.020 + tau_hi, fs, gain=0.8)
        aligned[pid] = buf

    low_band = LocalizationPipeline(f_min=800.0, f_max=2200.0, phat_beta=0.8)
    high_band = LocalizationPipeline(f_min=4300.0, f_max=5700.0, phat_beta=0.8)

    res_lo = low_band.localize(aligned, phones, room, kind="transient", fs=fs)
    res_hi = high_band.localize(aligned, phones, room, kind="transient", fs=fs)
    assert res_lo is not None and res_hi is not None

    assert math.hypot(res_lo.x_m - src_lo[0], res_lo.y_m - src_lo[1]) < 0.20
    assert math.hypot(res_hi.x_m - src_hi[0], res_hi.y_m - src_hi[1]) < 0.20
    assert (res_lo.x_m, res_lo.y_m) != (res_hi.x_m, res_hi.y_m)
