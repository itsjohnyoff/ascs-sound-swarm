"""Acoustic self-localization: the phones find each other by sound.

Every phone and the laptop take turns playing the calibration chirp while all phones record.
The schedule is a palindrome:

    laptop, p1, p2, ..., pN, pN, ..., p2, p1, laptop

Each phone hears every emission, including its own. Arrival times are read on each phone's own
clock, so nothing needs to be synced first. One joint least-squares fit recovers the phone
positions, every emission time, and each phone's clock offset and rate error. This is two-way
("BeepBeep") ranging generalised to N phones: the clock offsets cancel between a phone hearing
its own chirp and hearing the others' chirps, and the mirrored second half of the schedule
cancels clock-rate error to first order.

The laptop sits at a known point (the room centre), which fixes where the layout is. Turning
the layout around the laptop or mirroring it fits the sound equally well, so the orientation is
taken from the phones' current positions on the map (orthogonal Procrustes), and the user can
turn or mirror it afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from backend.dsp.clock import ChirpArrival

LAPTOP = "__laptop__"
SLOT_S = 0.7            # time between emissions
MATCH_TOL_S = 0.28      # how far an arrival may sit from its expected slot (network + output latency)
SELF_PATH_M = 0.03      # a phone's own speaker to its own microphone
MIN_PHONES = 3
STATUS_OK_M = 0.05      # per-phone fit error below this: trust the position
STATUS_CHECK_M = 0.20   # below this: usable, but ask the user to check it


def schedule(phone_ids: list[str]) -> list[str]:
    """Who emits in each slot: laptop, phones forward, phones backward, laptop."""
    ids = list(phone_ids)
    return [LAPTOP, *ids, *reversed(ids), LAPTOP]


def assign_slots(arrivals: list[ChirpArrival], fs: int, n_slots: int,
                 slot_s: float = SLOT_S, tol_s: float = MATCH_TOL_S) -> dict[int, ChirpArrival]:
    """Match one phone's detected chirps to schedule slots.

    The schedule's start time on this phone's clock is unknown, so try every (arrival, slot)
    pairing as the anchor and keep the alignment that explains the most arrivals, preferring
    louder ones on a tie.
    """
    times = [a.sample_index / fs for a in arrivals]
    best: tuple[int, float] = (0, -math.inf)
    best_map: dict[int, ChirpArrival] = {}
    for anchor_t in times:
        for anchor_slot in range(n_slots):
            start = anchor_t - anchor_slot * slot_s
            mapping: dict[int, ChirpArrival] = {}
            for slot in range(n_slots):
                expected = start + slot * slot_s
                near = [(abs(t - expected), a) for t, a in zip(times, arrivals) if abs(t - expected) <= tol_s]
                if near:
                    mapping[slot] = min(near, key=lambda item: item[0])[1]
            if len({id(a) for a in mapping.values()}) != len(mapping):
                continue  # one arrival claimed by two slots
            score = (len(mapping), sum(a.snr_db for a in mapping.values()))
            if score > best:
                best, best_map = score, mapping
    return best_map


def heard_from_audio(samples: np.ndarray, fs: int, start_sample_index: int, n_slots: int,
                     valid_mask: np.ndarray | None = None) -> dict[int, float]:
    """Detect the chirps in one phone's audio and return {slot: arrival time in local seconds}."""
    from backend.dsp.sync import detect_all_chirps

    arrivals = detect_all_chirps(samples, fs, start_sample_index=start_sample_index, valid_mask=valid_mask,
                                 min_spacing_s=SLOT_S / 2)
    return {slot: a.sample_index / fs for slot, a in assign_slots(arrivals, fs, n_slots).items()}


@dataclass
class PhoneFix:
    x_m: float
    y_m: float
    rms_m: float             # RMS fit error of every observation this phone takes part in
    heard: int               # emissions this phone heard
    heard_by: int            # listeners that heard this phone's emissions
    status: str              # ok | check | failed


@dataclass
class LayoutResult:
    success: bool
    phones: dict[str, PhoneFix] = field(default_factory=dict)
    distances_m: dict[str, float] = field(default_factory=dict)  # "a|b" -> fitted distance
    rms_m: float | None = None
    anchored: bool = False   # the laptop chirp fixed the position; otherwise the layout is centred
    error: str | None = None


def _classical_mds(d: np.ndarray) -> np.ndarray:
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    w, v = np.linalg.eigh(b)
    order = np.argsort(w)[::-1][:2]
    return v[:, order] * np.sqrt(np.maximum(w[order], 0.0))


def _procrustes_about(points: np.ndarray, target: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Rotate (and mirror if that fits better) points about pivot to best match target."""
    a, b = points - pivot, target - pivot
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ vt  # may include a reflection: both orientations explain the sound equally well
    return a @ r + pivot


def solve_layout(
    heard: dict[str, dict[int, float]],
    phone_ids: list[str],
    speaker_xy: tuple[float, float],
    c: float,
    *,
    prior: dict[str, tuple[float, float]] | None = None,
    self_path_m: float = SELF_PATH_M,
) -> LayoutResult:
    """Fit the layout.

    heard[listener][slot] is the arrival time in seconds on the listener's own clock of the
    emission in that slot of schedule(phone_ids).
    """
    ids = [p for p in phone_ids if p in heard]
    if len(ids) < MIN_PHONES:
        return LayoutResult(False, error=f"Need at least {MIN_PHONES} phones that heard the chirps")
    slots = schedule(phone_ids)
    n, n_slots = len(ids), len(slots)
    index = {p: i for i, p in enumerate(ids)}
    speaker = np.asarray(speaker_xy, dtype=float)

    # Observations: (listener index, slot, local time relative to that listener's mean).
    obs: list[tuple[int, int, float]] = []
    for p in ids:
        usable = {s: t for s, t in heard[p].items() if slots[s] == LAPTOP or slots[s] in index}
        if not usable:
            continue
        mean = float(np.mean(list(usable.values())))
        obs.extend((index[p], s, t - mean) for s, t in usable.items())
    emitted = sorted({s for _, s, _ in obs})
    laptop_heard = sum(1 for _, s, _ in obs if slots[s] == LAPTOP)

    # Starting guess: two-way distances per pair (clock offsets cancel), averaged over both halves.
    ids_all = list(phone_ids)

    def local(p: str, emitter: str, half: int) -> float | None:
        k = ids_all.index(emitter)
        return heard[p].get(1 + k if half == 0 else 2 * len(ids_all) - k)

    d0 = np.full((n, n), np.nan)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if j <= i:
                continue
            est = []
            for half in (0, 1):
                ta_a, ta_b, tb_b, tb_a = local(a, a, half), local(a, b, half), local(b, b, half), local(b, a, half)
                if None not in (ta_a, ta_b, tb_b, tb_a):
                    est.append(c / 2 * ((ta_b - ta_a) - (tb_b - tb_a)) + self_path_m)
            if est:
                d0[i, j] = d0[j, i] = max(0.05, float(np.mean(est)))
    np.fill_diagonal(d0, 0.0)
    if np.isnan(d0).all():
        return LayoutResult(False, error="The phones did not hear each other's chirps")
    filled = np.where(np.isnan(d0), np.nanmean(d0[d0 > 0]), d0)
    layout0 = _classical_mds(filled)
    layout0 -= layout0.mean(axis=0)

    # Unknowns: positions (2n), emission time per emitted slot except the first (gauge),
    # clock offset per phone, clock rate error per phone except the first (gauge).
    t_slots = emitted[1:]
    n_t = len(t_slots)
    slot_col = {s: k for k, s in enumerate(t_slots)}

    def unpack(x: np.ndarray):
        pos = x[: 2 * n].reshape(n, 2)
        t = np.zeros(n_slots)
        for s, k in slot_col.items():
            t[s] = x[2 * n + k]
        off = x[2 * n + n_t: 2 * n + n_t + n]
        eps = np.concatenate([[0.0], x[2 * n + n_t + n:]])
        return pos, t, off, eps

    li = np.array([o[0] for o in obs])
    sl = np.array([o[1] for o in obs])
    meas = np.array([o[2] for o in obs])
    emitter_idx = np.array([index.get(slots[s], -1) for s in sl])  # -1: laptop

    def path(pos: np.ndarray) -> np.ndarray:
        src = np.where(emitter_idx[:, None] >= 0, pos[np.maximum(emitter_idx, 0)], speaker)
        d = np.linalg.norm(src - pos[li], axis=1)
        return np.where(emitter_idx == li, self_path_m, d)

    def residuals(x: np.ndarray) -> np.ndarray:
        pos, t, off, eps = unpack(x)
        model = (1 + eps[li]) * (t[sl] + path(pos) / c) + off[li]
        return c * (meas - model)

    def initial(shift: np.ndarray) -> np.ndarray:
        pos = layout0 + speaker + shift
        d = path(pos) / c
        # Emission times and offsets are linear once positions are fixed.
        a = np.zeros((len(obs), n_t + n))
        for r, (lst, s) in enumerate(zip(li, sl)):
            if s in slot_col:
                a[r, slot_col[s]] = 1.0
            a[r, n_t + lst] = 1.0
        sol, *_ = np.linalg.lstsq(a, meas - d, rcond=None)
        return np.concatenate([pos.ravel(), sol, np.zeros(n - 1)])

    spread = float(np.max(np.abs(layout0))) or 1.0
    shifts = [np.zeros(2)] + [spread * 0.6 * np.array([math.cos(a), math.sin(a)])
                              for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    best = None
    for shift in shifts:
        fit = least_squares(residuals, initial(shift), method="trf", loss="soft_l1", f_scale=0.05, max_nfev=400)
        if best is None or fit.cost < best.cost:
            best = fit
    pos, *_ = unpack(best.x)
    res = residuals(best.x)

    anchored = laptop_heard >= MIN_PHONES
    if not anchored:
        pos = pos - pos.mean(axis=0) + speaker  # without the laptop chirp, centre the layout
    if prior and all(p in prior for p in ids):
        target = np.array([prior[p] for p in ids], dtype=float)
        pos = _procrustes_about(pos, target, speaker if anchored else pos.mean(axis=0))

    fixes: dict[str, PhoneFix] = {}
    for p, i in index.items():
        mine = (li == i) | (emitter_idx == i)
        rms = float(np.sqrt(np.mean(res[mine] ** 2))) if mine.any() else math.inf
        heard_n = int(np.sum(li == i))
        heard_by = int(len({int(x) for x in li[(emitter_idx == i) & (li != i)]}))
        status = "ok" if rms < STATUS_OK_M else "check" if rms < STATUS_CHECK_M else "failed"
        if heard_by < 2 or heard_n < 3:
            status = "failed"
        fixes[p] = PhoneFix(float(pos[i, 0]), float(pos[i, 1]), rms, heard_n, heard_by, status)
    distances = {f"{a}|{b}": float(np.linalg.norm(pos[index[a]] - pos[index[b]]))
                 for k, a in enumerate(ids) for b in ids[k + 1:]}
    overall = float(np.sqrt(np.mean(res ** 2)))
    ok = sum(f.status != "failed" for f in fixes.values()) >= MIN_PHONES
    return LayoutResult(ok, fixes, distances, overall, anchored,
                        None if ok else "Too few phones could be placed. Turn the volume up and try again.")
