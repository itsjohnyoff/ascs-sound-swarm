"""Pure-numpy 2D room simulator writing CONTRACTS 8 recordings.

Direct path plus first-order wall images, exact fractional delays (windowed sinc),
per-phone sample rate, start offset, clock drift, noise, drops and clipping.

Phone clock model (ground truth): local sample k of the recording happens at true time
    t = start_s + k / (fs * (1 + drift_ppm * 1e-6))
and carries worklet sample_index = start_index + k.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from backend.server.judge_eval import inside_hull
from shared.protocol import RoomConfig, speed_of_sound
from sim.sources import SRC_FS, render

TAPS = 16  # half-width of the windowed-sinc kernel


@dataclass
class SimPhone:
    phone_id: str
    x_m: float
    y_m: float
    fs: int = 48000
    start_s: float = 0.0          # true time of the phone's first recorded sample
    drift_ppm: float = 0.0
    snr_db: float = 40.0          # vs. a direct-path chirp at this phone
    start_index: int = 0
    drop_ranges_s: list[tuple[float, float]] = field(default_factory=list)  # local seconds
    clip_level: float | None = None
    # Occlusion: direct path scaled by direct_gain, plus a reflection (delay_s, gain) of the same source
    direct_gain: float = 1.0
    occluder_echo: tuple[float, float] | None = None


@dataclass
class SimEvent:
    t_s: float                    # true emission time
    x_m: float
    y_m: float
    kind: str                     # chirp | clap | drone | speech
    duration_s: float | None = None
    gain: float = 1.0


def _paths(x, y, room: RoomConfig, reflection_gain: float, echo: tuple[float, float] | None):
    """(source_xy, gain, extra_delay_s) for the direct path and its images."""
    out = [((x, y), 1.0, 0.0)]
    if echo is not None:
        out.append(((x, y), echo[1], echo[0]))
    if reflection_gain > 0:
        w, h = room.width_m, room.height_m
        for img in ((-x, y), (2 * w - x, y), (x, -y), (x, 2 * h - y)):
            out.append((img, reflection_gain, 0.0))
    return out


def _sinc_sample(src: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Evaluate band-limited src (at SRC_FS) at fractional sample positions pos."""
    base = np.floor(pos).astype(np.int64)
    k = base[:, None] + np.arange(-TAPS + 1, TAPS + 1)[None, :]
    frac = pos[:, None] - k
    win = 0.5 * (1 + np.cos(np.pi * np.clip(frac / TAPS, -1, 1)))
    kern = np.sinc(frac) * win
    valid = (k >= 0) & (k < len(src))
    vals = np.where(valid, src[np.clip(k, 0, len(src) - 1)], 0.0)
    return np.sum(vals * kern, axis=1)


def simulate(
    room: RoomConfig,
    phones: list[SimPhone],
    events: list[SimEvent],
    duration_s: float,
    *,
    reflection_gain: float = 0.0,
    echo: tuple[float, float] | None = None,
    seed: int = 0,
    notes: str = "",
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = speed_of_sound(room.temperature_c)
    sources = [render(e.kind, np.random.default_rng(seed * 1000 + i), e.duration_s) * e.gain
               for i, e in enumerate(events)]
    arrays: dict[str, np.ndarray] = {}
    truth_phones = {}
    for ph in phones:
        n = int(round(duration_s * ph.fs))
        eff_fs = ph.fs * (1 + ph.drift_ppm * 1e-6)
        t_true = ph.start_s + np.arange(n) / eff_fs
        pcm = np.zeros(n, dtype=np.float64)
        for ev, src in zip(events, sources):
            # Sum every path into one fractional-delay FIR at SRC_FS, then resample once
            # onto this phone's (drifting) clock.
            paths = [(float(np.hypot(sx - ph.x_m, sy - ph.y_m)), g, extra)
                     for (sx, sy), g, extra in _paths(ev.x_m, ev.y_m, room, reflection_gain, echo)]
            d_direct = paths[0][0]
            paths[0] = (d_direct, paths[0][1] * ph.direct_gain, 0.0)
            if ph.occluder_echo is not None:
                paths.append((d_direct, ph.occluder_echo[1], ph.occluder_echo[0]))
            delays = [d / c + extra for d, _, extra in paths]
            t_ref = ev.t_s + min(delays)
            span = int(np.ceil((max(delays) - min(delays)) * SRC_FS))
            k = np.arange(span + 2 * TAPS + 1, dtype=np.float64)
            fir = np.zeros_like(k)
            for (d, g, _), delay in zip(paths, delays):
                frac = k - ((delay - min(delays)) * SRC_FS + TAPS)
                win = 0.5 * (1 + np.cos(np.pi * np.clip(frac / TAPS, -1, 1)))
                fir += g / max(d, 0.5) * np.sinc(frac) * win
            y = fftconvolve(src, fir)
            lo = np.searchsorted(t_true, t_ref - 2 * TAPS / SRC_FS)
            hi = np.searchsorted(t_true, t_ref + len(y) / SRC_FS)
            if hi > lo:
                pcm[lo:hi] += _sinc_sample(y, (t_true[lo:hi] - t_ref) * SRC_FS + TAPS)
        d_spk = max(float(np.hypot(room.speaker_x_m - ph.x_m, room.speaker_y_m - ph.y_m)), 0.5)
        ref_rms = 0.85 / np.sqrt(2) / d_spk
        pcm += rng.standard_normal(n) * ref_rms * 10 ** (-ph.snr_db / 20)
        if ph.clip_level is not None:
            pcm = np.clip(pcm, -ph.clip_level, ph.clip_level)
        valid = np.ones(n, dtype=bool)
        for a, b in ph.drop_ranges_s:
            sl = slice(int(a * ph.fs), int(b * ph.fs))
            valid[sl] = False
            pcm[sl] = 0.0
        pid = ph.phone_id
        arrays[f"{pid}_pcm"] = pcm.astype(np.float32)
        arrays[f"{pid}_start_index"] = np.asarray(ph.start_index, dtype=np.int64)
        arrays[f"{pid}_fs"] = np.asarray(ph.fs, dtype=np.int64)
        arrays[f"{pid}_valid"] = valid
        truth_phones[pid] = {"start_s": ph.start_s, "drift_ppm": ph.drift_ppm, "fs": ph.fs,
                             "start_index": ph.start_index, "snr_db": ph.snr_db}

    hull = [(p.x_m, p.y_m) for p in phones]
    meta = {
        "room": room.model_dump(mode="json"),
        "phones": {p.phone_id: {"phone_id": p.phone_id, "label": p.phone_id, "x_m": p.x_m, "y_m": p.y_m}
                   for p in phones},
        "speaker": {"x_m": room.speaker_x_m, "y_m": room.speaker_y_m},
        "temperature_c": room.temperature_c,
        "chirp_times_s": [e.t_s for e in events if e.kind == "chirp"],
        "notes": notes or "simulated",
        "simulated": True,
        "truth": {
            "phones": truth_phones,
            "events": [{**asdict(e), "inside_hull": inside_hull(e.x_m, e.y_m, hull)} for e in events],
            "reflection_gain": reflection_gain,
            "echo": list(echo) if echo else None,
        },
    }
    arrays["meta"] = np.asarray(json.dumps(meta))
    return arrays


def true_time(truth_phone: dict, sample_index: float) -> float:
    """Ground-truth emission-clock time of a phone's worklet sample index."""
    k = sample_index - truth_phone["start_index"]
    return truth_phone["start_s"] + k / (truth_phone["fs"] * (1 + truth_phone["drift_ppm"] * 1e-6))


def save_npz(arrays: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
DEFAULT_ROOM = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=22.0, speaker_x_m=4.0, speaker_y_m=0.4)


def default_phones(rng: np.random.Generator, n: int = 4, *, offsets_s: float = 0.3, drift_ppm: float = 50.0,
                   snr_db: float = 30.0) -> list[SimPhone]:
    spots = [(1.0, 1.0), (7.0, 1.2), (6.8, 5.0), (1.2, 4.8), (4.0, 5.6), (0.6, 3.0)]
    rates = [48000, 44100, 48000, 44100, 48000, 44100]
    return [
        SimPhone(f"p{i + 1}", *spots[i], fs=rates[i],
                 start_s=float(rng.uniform(-offsets_s, offsets_s)),
                 drift_ppm=float(rng.uniform(-drift_ppm, drift_ppm)),
                 snr_db=snr_db, start_index=int(rng.integers(0, 2_000_000)))
        for i in range(n)
    ]


def sync_scenario(*, n_phones: int = 4, n_chirps: int = 3, chirp_spacing_s: float = 30.0, snr_db: float = 15.0,
                  echo: tuple[float, float] | None = (0.007, 0.65), seed: int = 0,
                  room: RoomConfig = DEFAULT_ROOM) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    phones = default_phones(rng, n_phones, snr_db=snr_db)
    events = [SimEvent(1.0 + k * chirp_spacing_s, room.speaker_x_m, room.speaker_y_m, "chirp")
              for k in range(n_chirps)]
    dur = 1.0 + (n_chirps - 1) * chirp_spacing_s + 1.5
    return simulate(room, phones, events, dur, echo=echo, seed=seed, notes="sync_scenario")


CENTRED_ROOM = RoomConfig(width_m=8.0, height_m=6.0, temperature_c=22.0, speaker_x_m=4.0, speaker_y_m=3.0)


def locate_scenario(*, n_phones: int = 4, snr_db: float = 30.0, reflection_gain: float = 0.3, seed: int = 0,
                    room: RoomConfig = CENTRED_ROOM, slot_s: float = 0.7, jitter_s: float = 0.08,
                    self_path_m: float = 0.03) -> dict[str, np.ndarray]:
    """Self-localization run: laptop, every phone forward, every phone backward, laptop.

    Each emission starts late by a random amount (network and output latency), and each phone's
    speaker sits self_path_m from its own microphone.
    """
    from backend.dsp.ranging import LAPTOP, schedule

    rng = np.random.default_rng(seed)
    phones = default_phones(rng, n_phones, snr_db=snr_db)
    by_id = {p.phone_id: p for p in phones}
    events = []
    for k, who in enumerate(schedule([p.phone_id for p in phones])):
        t = 1.0 + k * slot_s + float(rng.uniform(0, jitter_s))
        if who == LAPTOP:
            events.append(SimEvent(t, room.speaker_x_m, room.speaker_y_m, "chirp"))
        else:
            ph = by_id[who]
            events.append(SimEvent(t, ph.x_m + self_path_m, ph.y_m, "chirp"))
    dur = events[-1].t_s + 1.5
    return simulate(room, phones, events, dur, reflection_gain=reflection_gain, seed=seed, notes="locate_scenario")


JUDGE_POINTS = [  # (x, y, kind); last two are outside the phone hull
    (3.0, 2.5, "clap"), (5.5, 3.5, "drone"), (2.0, 4.0, "clap"), (4.5, 1.8, "drone"),
    (6.0, 2.2, "clap"), (3.5, 4.2, "drone"), (4.0, 3.0, "clap"), (2.2, 2.0, "drone"),
    (0.4, 2.8, "clap"), (7.6, 3.4, "drone"),
]


def judge_scenario(*, n_phones: int = 4, snr_db: float = 30.0, reflection_gain: float = 0.3, seed: int = 0,
                   room: RoomConfig = DEFAULT_ROOM, event_gap_s: float = 2.5,
                   chirp_times_s: tuple[float, ...] = (1.0, 11.0),
                   occluded: tuple[str, float, float, float] | None = None) -> dict[str, np.ndarray]:
    """Chirps, then 10 positions (5 claps, 5 drones), 2 of them outside the hull.

    occluded=(phone_id, direct_gain, echo_delay_s, echo_gain) blocks one phone's direct path
    for the target sounds (not the chirps), which corrupts every pair that phone is in.
    """
    rng = np.random.default_rng(seed)
    phones = default_phones(rng, n_phones, snr_db=snr_db)
    events = [SimEvent(t, room.speaker_x_m, room.speaker_y_m, "chirp") for t in chirp_times_s]
    t = chirp_times_s[-1] + 3.0
    for x, y, kind in JUDGE_POINTS:
        events.append(SimEvent(t, x, y, kind, duration_s=0.9 if kind == "drone" else None))
        t += event_gap_s
    rec = simulate(room, phones, events, t + 1.0, reflection_gain=reflection_gain, seed=seed, notes="judge_scenario")
    if occluded is None:
        return rec
    # Re-render the occluded phone: chirps clean, target sounds through the occluder.
    pid, g_direct, delay, g_echo = occluded
    ph = next(p for p in phones if p.phone_id == pid)
    chirps = [e for e in events if e.kind == "chirp"]
    targets = [e for e in events if e.kind != "chirp"]
    clean = simulate(room, [ph], chirps, t + 1.0, reflection_gain=reflection_gain, seed=seed)
    ph.direct_gain, ph.occluder_echo, ph.snr_db = g_direct, (delay, g_echo), 200.0
    blocked = simulate(room, [ph], targets, t + 1.0, reflection_gain=reflection_gain, seed=seed)
    rec[f"{pid}_pcm"] = (clean[f"{pid}_pcm"] + blocked[f"{pid}_pcm"]).astype(np.float32)
    meta = json.loads(str(rec["meta"]))
    meta["truth"]["occluded"] = list(occluded)
    rec["meta"] = np.asarray(json.dumps(meta))
    return rec
