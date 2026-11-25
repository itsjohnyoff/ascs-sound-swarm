"""Is ggwave a timing anchor? python -m analysis.ggwave_eval [--trials 30]

Sends the same event as a ggwave packet and as the LFM chirp to 3 simulated phones at known
positions, with noise. Compares the error of the inter-phone time difference (TDOA) each
method gives: ggwave by the moment the streaming decoder reports the payload, LFM by the
matched-filter arrival (backend.dsp.chirp_detect). Needs `ggwave` (pip install ggwave; on
Python 3.13 build it with GGWAVE_USE_CYTHON=1 and a current setuptools).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from backend.dsp.chirp_detect import detect_chirps
from shared.chirp import generate_lfm_chirp
from shared.protocol import speed_of_sound
from sim.room import DEFAULT_ROOM, _sinc_sample

FS = 48000
CHUNK = 1024           # ggwave ignores feeds smaller than its 1024-sample frame
T0 = 0.5               # true emission time, s
DURATION = 3.0
PHONES = {"p1": (1.0, 1.0), "p2": (7.0, 1.2), "p3": (4.0, 5.6)}


def _render(src: np.ndarray, xy: tuple[float, float], rng: np.random.Generator, snr_db: float) -> tuple[np.ndarray, float]:
    c = speed_of_sound(DEFAULT_ROOM.temperature_c)
    d = math.hypot(xy[0] - DEFAULT_ROOM.speaker_x_m, xy[1] - DEFAULT_ROOM.speaker_y_m)
    delay = T0 + d / c
    n = int(DURATION * FS)
    pos = (np.arange(n) / FS - delay) * FS
    y = np.zeros(n)
    lo = max(0, int(delay * FS) - 40)
    hi = min(n, int(delay * FS) + len(src) + 40)
    y[lo:hi] = _sinc_sample(src.astype(np.float64), pos[lo:hi])
    y += rng.standard_normal(n) * np.sqrt(np.mean(src ** 2)) * 10 ** (-snr_db / 20)
    return y.astype(np.float32), delay


def _ggwave_decode_time(pcm: np.ndarray, ggwave, instance) -> float | None:
    for k in range(0, len(pcm), CHUNK):
        out = ggwave.decode(instance, pcm[k:k + CHUNK].astype("<f4").tobytes())
        if out is not None:
            return (k + CHUNK) / FS
    return None


def evaluate(trials: int = 30, snr_db: float = 15.0, seed: int = 0) -> dict:
    import ggwave

    rng = np.random.default_rng(seed)
    wave = np.frombuffer(ggwave.encode("C01", protocolId=1, volume=50), dtype="<f4")
    chirp = generate_lfm_chirp(FS)
    c = speed_of_sound(DEFAULT_ROOM.temperature_c)
    names = list(PHONES)
    err = {"ggwave": [], "lfm": []}
    decoded = total = 0
    for _ in range(trials):
        t_gg, t_lfm, true_end, true_start = {}, {}, {}, {}
        for pid in names:
            pcm, delay = _render(wave, PHONES[pid], rng, snr_db)
            instance = ggwave.init()  # fresh decoder per phone, state leaks between streams
            td = _ggwave_decode_time(pcm, ggwave, instance)
            ggwave.free(instance)
            total += 1
            if td is not None:
                decoded += 1
                t_gg[pid] = td
            true_end[pid] = delay + len(wave) / FS
            pcm_c, delay_c = _render(chirp, PHONES[pid], rng, snr_db)
            arr = detect_chirps(pcm_c, FS)
            if arr:
                t_lfm[pid] = arr[0].sample_index / FS
            true_start[pid] = delay_c
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                pa, pb = names[a], names[b]
                if pa in t_gg and pb in t_gg:
                    err["ggwave"].append((t_gg[pa] - t_gg[pb]) - (true_end[pa] - true_end[pb]))
                if pa in t_lfm and pb in t_lfm:
                    err["lfm"].append((t_lfm[pa] - t_lfm[pb]) - (true_start[pa] - true_start[pb]))

    def rms_ms(v):
        return float(np.sqrt(np.mean(np.square(v))) * 1e3) if v else float("nan")

    out = {"trials": trials, "snr_db": snr_db, "decode_rate": decoded / total, "chunk_ms": CHUNK / FS * 1e3,
           "waveform_s": len(wave) / FS}
    for k in ("ggwave", "lfm"):
        out[f"{k}_tdoa_rms_ms"] = rms_ms(err[k])
        out[f"{k}_tdoa_rms_cm"] = out[f"{k}_tdoa_rms_ms"] * 1e-3 * c * 100
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--snr", type=float, nargs="+", default=[20.0, 10.0, 0.0])
    args = ap.parse_args(argv)
    for snr in args.snr:
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in evaluate(args.trials, snr).items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
