"""TDD tests for 2D SRP-PHAT + cycle-consistent Huber multilateration & LocalizationPipeline."""

from __future__ import annotations

import math
import time
import numpy as np
from scipy.ndimage import map_coordinates

from shared.protocol import PhoneConfig, RoomConfig, speed_of_sound
from backend.dsp.pipeline import LocalizationPipeline


def _make_room_and_phones() -> tuple[RoomConfig, dict[str, PhoneConfig]]:
    room = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=20.0, speaker_x_m=4.0, speaker_y_m=0.5)
    coords = [("p1", 0.5, 0.5), ("p2", 7.5, 0.5), ("p3", 7.5, 5.5), ("p4", 0.5, 5.5)]
    phones = {pid: PhoneConfig(phone_id=pid, label=pid, x_m=x, y_m=y, sample_rate=48000) for pid, x, y in coords}
    return room, phones


def _simulate_array_event(
    src_xy: tuple[float, float],
    source_sig: np.ndarray,
    phones: dict[str, PhoneConfig],
    room: RoomConfig,
    fs: int = 48000,
    snr_db: float = 18.0,
    echo_amp: float = 0.35,
    echo_delay_s: float = 0.008,
    corrupt_phone: str | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = speed_of_sound(room.temperature_c)
    n = len(source_sig)
    idx = np.arange(n, dtype=np.float64)

    out: dict[str, np.ndarray] = {}
    sig_rms = float(np.sqrt(np.mean(source_sig * source_sig))) + 1e-12
    noise_std = sig_rms / (10.0 ** (snr_db / 20.0))

    for pid, pcfg in phones.items():
        dist = math.hypot(src_xy[0] - pcfg.x_m, src_xy[1] - pcfg.y_m)
        tau = dist / c
        # Slight per-phone echo variation simulating wall reflections
        local_echo_delay = echo_delay_s + 0.0015 * ((hash(pid) % 5) - 2)
        direct = map_coordinates(source_sig.astype(np.float64), [idx - tau * fs], order=3, mode="constant", cval=0.0)
        echo = echo_amp * map_coordinates(
            source_sig.astype(np.float64),
            [idx - (tau + max(0.003, local_echo_delay)) * fs],
            order=3,
            mode="constant",
            cval=0.0,
        )
        if corrupt_phone == pid:
            # Realistic multipath corruption: a +16 ms wall reflection is 1.5x stronger than direct path,
            # causing global GCC-PHAT peaks on all 3 pairs involving this phone to lock onto the echo.
            direct = direct + 1.5 * map_coordinates(
                source_sig.astype(np.float64), [idx - (tau + 0.016) * fs], order=3, mode="constant", cval=0.0
            )
        y = (direct + echo + rng.normal(0.0, noise_std, size=n)).astype(np.float32)
        out[pid] = y
    return out


def _make_clap(fs: int = 48000, duration_s: float = 0.5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    sig = np.zeros(n, dtype=np.float32)
    burst_len = int(round(0.025 * fs))
    t = np.arange(burst_len, dtype=np.float64) / fs
    burst = np.exp(-t / 0.0045) * rng.normal(0.0, 1.0, size=burst_len)
    s0 = int(round(0.15 * fs))
    sig[s0 : s0 + burst_len] = (0.7 * burst / (np.max(np.abs(burst)) + 1e-12)).astype(np.float32)
    return sig


def _make_drone_buzz(fs: int = 48000, duration_s: float = 0.6, f0: float = 140.0, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    harmonics = sum((0.18 / k) * np.sin(2.0 * np.pi * f0 * k * t + 0.3 * k) for k in range(1, 9))
    rotor_noise = 0.04 * rng.normal(0.0, 1.0, size=n)
    return (harmonics + rotor_noise).astype(np.float32)


def test_10_judge_positions_claps_and_drones_accuracy_and_speed() -> None:
    room, phones = _make_room_and_phones()
    pipeline = LocalizationPipeline()
    fs = 48000

    positions = [
        (2.0, 1.5),
        (4.0, 3.0),
        (6.2, 2.1),
        (3.1, 4.4),
        (5.5, 4.8),
        (1.6, 3.7),
        (6.4, 4.2),
        (4.3, 1.4),
        (2.8, 2.6),
        (5.1, 3.3),
    ]

    errors_m: list[float] = []
    runtimes_ms: list[float] = []

    for idx, pos in enumerate(positions):
        if idx < 5:
            src_sig = _make_clap(fs=fs, seed=100 + idx)
            kind_lbl = "clap"
        else:
            src_sig = _make_drone_buzz(fs=fs, f0=95.0 + 30.0 * (idx - 5), seed=200 + idx)
            kind_lbl = "drone"

        arr = _simulate_array_event(pos, src_sig, phones, room, fs=fs, snr_db=16.0, echo_amp=0.35, seed=300 + idx)

        t0 = time.perf_counter()
        res = pipeline.detect_and_localize(arr, phones, room, fs=fs, timestamp_s=10.0 + idx)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        runtimes_ms.append(dt_ms)

        assert res is not None, f"Missed detection at pos {idx} ({kind_lbl})"
        assert res.heatmap is not None
        assert res.heatmap.nx == 32 and res.heatmap.ny == 24
        assert len(res.heatmap.values) == 32 * 24
        err_m = math.hypot(res.x_m - pos[0], res.y_m - pos[1])
        errors_m.append(err_m)
        print(
            f"\n  Pos {idx+1:2d} ({kind_lbl:5s}): gt={pos} est=({res.x_m:.2f}, {res.y_m:.2f}) "
            f"err={err_m:.3f} m res={res.residual_ms:.3f} ms unc={res.uncertainty_m:.2f} m ({dt_ms:.1f} ms)"
        )

    mean_err = float(np.mean(errors_m))
    mean_rt = float(np.mean(runtimes_ms))
    print(f"\n10-position summary: mean_err={mean_err:.3f} m, max_err={max(errors_m):.3f} m, mean_runtime={mean_rt:.1f} ms")
    assert mean_err < 0.15
    assert mean_rt < 50.0


def test_outlier_pair_corruption_remains_accurate() -> None:
    room, phones = _make_room_and_phones()
    pipeline = LocalizationPipeline()
    fs = 48000

    positions = [(2.5, 2.0), (5.5, 2.2), (3.5, 4.2), (6.0, 4.0)]
    errors_m: list[float] = []
    for idx, pos in enumerate(positions):
        src_sig = _make_clap(fs=fs, seed=400 + idx)
        arr = _simulate_array_event(
            pos,
            src_sig,
            phones,
            room,
            fs=fs,
            snr_db=16.0,
            echo_amp=0.30,
            corrupt_phone="p2",
            seed=500 + idx,
        )
        res = pipeline.detect_and_localize(arr, phones, room, fs=fs, timestamp_s=50.0 + idx)
        assert res is not None
        err = math.hypot(res.x_m - pos[0], res.y_m - pos[1])
        errors_m.append(err)
        print(f"\n  Outlier pos {idx+1}: gt={pos} est=({res.x_m:.2f}, {res.y_m:.2f}) err={err:.3f} m")

    mean_err = float(np.mean(errors_m))
    print(f"\nCorrupted-phone mean error: {mean_err:.3f} m")
    assert mean_err < 0.35


def test_two_phone_diagnostic_mode_and_silence() -> None:
    room, phones = _make_room_and_phones()
    two_phones = {"p1": phones["p1"], "p2": phones["p2"]}
    pipeline = LocalizationPipeline()
    fs = 48000

    # Quiet returns None
    quiet = {pid: np.zeros(int(0.4 * fs), dtype=np.float32) for pid in two_phones}
    assert pipeline.detect_and_localize(quiet, two_phones, room, fs=fs) is None

    # 2-phone clap does not crash and produces valid result + heatmap
    clap = _make_clap(fs=fs, seed=99)
    arr = _simulate_array_event((3.2, 2.5), clap, two_phones, room, fs=fs, echo_amp=0.0, seed=98)
    res = pipeline.detect_and_localize(arr, two_phones, room, fs=fs, timestamp_s=1.0)
    assert res is not None
    assert res.heatmap is not None and len(res.heatmap.values) == 32 * 24
    assert len(res.pair_tdoas_ms) == 1
