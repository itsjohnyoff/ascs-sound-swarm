"""Sub-millisecond chirp synchronization, first-arrival echo defense, and PPM drift alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import map_coordinates
from scipy.signal import butter, fftconvolve, find_peaks, hilbert, sosfiltfilt

from shared.protocol import (
    CHIRP_DURATION_S,
    DUAL_CHIRP_GAP_S,
    CalibrationMode,
    CalibrationResult,
    PhoneAudioSlice,
    PhoneConfig,
    PhoneSyncReport,
    RoomConfig,
    speed_of_sound,
)
from backend.dsp.chirp import generate_lfm_chirp
from backend.dsp.clock import (
    ChirpArrival,
    ClockModel,
    ReferenceClockFitter,
    flight_time_s,
)
from backend.dsp.gcc_phat import compute_gcc_phat


def _parabolic_refine(arr: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    y_m1 = float(arr[idx - 1])
    y_0 = float(arr[idx])
    y_p1 = float(arr[idx + 1])
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-18:
        return float(idx)
    delta = float(np.clip(0.5 * (y_m1 - y_p1) / denom, -0.5, 0.5))
    return float(idx) + delta


def _local_template_ncc(
    env: np.ndarray,
    x_filt: np.ndarray,
    idx: int,
    tpl_len: int,
) -> float:
    """Normalized correlation magnitude against unit-norm chirp template at idx."""
    seg = x_filt[idx : idx + tpl_len]
    local_norm = float(np.linalg.norm(seg))
    if local_norm <= 1e-12:
        return 0.0
    return float(env[idx]) / local_norm


def _precompute_template_halves(tpl: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Precompute unit-norm analytic templates for first and second chirp frequency halves."""
    if len(tpl) < 64:
        return None
    h = len(tpl) // 2
    t1 = hilbert(tpl[:h]) / (float(np.linalg.norm(tpl[:h])) + 1e-12)
    t2 = hilbert(tpl[h:]) / (float(np.linalg.norm(tpl[h:])) + 1e-12)
    return t1, t2


def _split_half_min_ncc(
    x_filt: np.ndarray,
    tpl_halves: tuple[np.ndarray, np.ndarray],
    idx: int,
) -> float:
    """Minimum phase-invariant NCC across both chirp halves (rejects impulsive claps)."""
    t1, t2 = tpl_halves
    h = len(t1)
    L = h + len(t2)
    if idx < 0 or idx + L > len(x_filt):
        return 0.0
    seg1 = x_filt[idx : idx + h]
    seg2 = x_filt[idx + h : idx + L]
    n1 = float(np.linalg.norm(seg1))
    n2 = float(np.linalg.norm(seg2))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    return min(abs(complex(np.dot(seg1, t1))) / n1, abs(complex(np.dot(seg2, t2))) / n2)


def _valid_envelope_mask(
    valid_mask: np.ndarray | None,
    n_samples: int,
    tpl_len: int,
) -> np.ndarray | None:
    """Return boolean mask over matched-filter envelope indices where the full template is valid."""
    if valid_mask is None:
        return None
    vm = np.asarray(valid_mask, dtype=bool)
    if len(vm) != n_samples:
        raise ValueError(f"valid_mask length ({len(vm)}) != samples length ({n_samples})")
    if bool(vm.all()):
        return None
    invalid_prefix = np.concatenate(([0], np.cumsum(~vm)))
    return invalid_prefix[tpl_len:] == invalid_prefix[:-tpl_len]


def _compute_chirp_envelope(
    samples_f32: np.ndarray,
    fs: int,
    template: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    f_low: float = 1200.0,
    f_high: float = 8500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None:
    """Bandpass filter audio and compute matched-filter Hilbert envelope."""
    if valid_mask is not None and len(valid_mask) != len(samples_f32):
        raise ValueError(f"valid_mask length ({len(valid_mask)}) != samples length ({len(samples_f32)})")
    tpl = (generate_lfm_chirp(fs=fs) if template is None else np.asarray(template)).astype(np.float64)
    if len(samples_f32) < len(tpl) + 16:
        return None

    tpl_norm = float(np.linalg.norm(tpl))
    if tpl_norm < 1e-12:
        return None
    tpl /= tpl_norm

    hi = min(f_high, 0.48 * fs)
    sos = butter(4, [f_low, hi], btype="bandpass", fs=fs, output="sos")
    x_filt = sosfiltfilt(sos, samples_f32.astype(np.float64))

    corr = fftconvolve(x_filt, tpl[::-1], mode="valid")
    n_fast = next_fast_len(len(corr))
    env = np.abs(hilbert(corr, N=n_fast)[: len(corr)])

    env_valid = _valid_envelope_mask(valid_mask, len(samples_f32), len(tpl))
    if env_valid is not None:
        if not bool(np.any(env_valid)):
            return None
        # Estimate Rayleigh noise floor strictly over valid envelope samples so large zero-filled
        # packet gaps cannot collapse sigma toward zero and trigger false chirp detections.
        sigma = float(np.median(env[env_valid])) / 1.17741 + 1e-12
        env = np.where(env_valid, env, 0.0)
    else:
        sigma = float(np.median(env)) / 1.17741 + 1e-12

    return env, x_filt, tpl, sigma, len(tpl)


def _pick_first_arrival_with_ncc(
    env: np.ndarray,
    x_filt: np.ndarray,
    tpl_len: int,
    fs: int,
    start: int,
    end: int,
    sigma: float,
    min_snr_db: float,
    min_ncc: float,
    lookback_s: float = 0.025,
    direct_path_ratio: float = 0.55,
    tpl_halves: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float | None, float, float]:
    """Locate earliest direct-path peak in env[start:end] matching the chirp template."""
    start = max(0, start)
    end = min(len(env), end)
    if end - start < 3:
        return None, -100.0, 0.0

    region = env[start:end]
    rel_max = int(np.argmax(region))
    i_max = start + rel_max
    peak_val = float(env[i_max])
    snr_db = 20.0 * math.log10(max(peak_val, 1e-12) / max(sigma, 1e-12))
    if snr_db < min_snr_db:
        return None, snr_db, 0.0

    half_thresh = 0.5 * min_ncc
    # Search backwards before strongest peak to recover direct path even if a wall echo is louder
    lookback = int(round(lookback_s * fs))
    win_start = max(start, i_max - lookback)
    sub = env[win_start : i_max + 1]
    peaks, _ = find_peaks(sub, height=direct_path_ratio * peak_val, distance=max(1, int(0.001 * fs)))
    for pk in peaks:
        cand = win_start + int(pk)
        ncc = _local_template_ncc(env, x_filt, cand, tpl_len)
        if ncc >= min_ncc and (tpl_halves is None or _split_half_min_ncc(x_filt, tpl_halves, cand) >= half_thresh):
            return _parabolic_refine(env, cand), snr_db, ncc

    ncc_max = _local_template_ncc(env, x_filt, i_max, tpl_len)
    if ncc_max >= min_ncc and (tpl_halves is None or _split_half_min_ncc(x_filt, tpl_halves, i_max) >= half_thresh):
        return _parabolic_refine(env, i_max), snr_db, ncc_max
    return None, snr_db, ncc_max


def _pick_first_arrival(
    env: np.ndarray,
    x_filt: np.ndarray,
    tpl_len: int,
    fs: int,
    start: int,
    end: int,
    sigma: float,
    min_snr_db: float,
    min_ncc: float,
    lookback_s: float = 0.025,
    direct_path_ratio: float = 0.55,
    tpl_halves: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float | None, float]:
    idx, snr_db, _ = _pick_first_arrival_with_ncc(
        env=env,
        x_filt=x_filt,
        tpl_len=tpl_len,
        fs=fs,
        start=start,
        end=end,
        sigma=sigma,
        min_snr_db=min_snr_db,
        min_ncc=min_ncc,
        lookback_s=lookback_s,
        direct_path_ratio=direct_path_ratio,
        tpl_halves=tpl_halves,
    )
    return idx, snr_db


CLIP_THRESHOLD: float = 0.99
MIN_CLIPPED_SAMPLES: int = 2
DEFAULT_CHIRP_MIN_NCC: float = 0.15


def is_window_clipped(
    samples: np.ndarray,
    clip_threshold: float = CLIP_THRESHOLD,
    min_clipped_samples: int = MIN_CLIPPED_SAMPLES,
) -> bool:
    """Return True when at least min_clipped_samples reach clip_threshold."""
    if len(samples) == 0:
        return False
    return int(np.count_nonzero(np.abs(samples) >= clip_threshold)) >= min_clipped_samples


def detect_all_chirps(
    samples_f32: np.ndarray,
    fs: int,
    template: np.ndarray | None = None,
    start_sample_index: int = 0,
    valid_mask: np.ndarray | None = None,
    min_spacing_s: float = 1.0,
    min_snr_db: float = 12.0,
    min_ncc: float = DEFAULT_CHIRP_MIN_NCC,
    lookback_s: float = 0.025,
    direct_path_ratio: float = 0.55,
    f_low: float = 1200.0,
    f_high: float = 8500.0,
) -> list[ChirpArrival]:
    """Detect all LFM chirps in a recording or buffer slice, returning chronological Arrivals."""
    prep = _compute_chirp_envelope(
        samples_f32,
        fs=fs,
        template=template,
        valid_mask=valid_mask,
        f_low=f_low,
        f_high=f_high,
    )
    if prep is None:
        return []
    env, x_filt, tpl, sigma, tpl_len = prep
    tpl_halves = _precompute_template_halves(tpl)

    min_height = sigma * (10.0 ** (min_snr_db / 20.0))
    cand_dist = max(1, int(round(min(0.05, 0.5 * min_spacing_s) * fs)))
    cand_pks, _ = find_peaks(env, height=min_height, distance=cand_dist)
    if len(cand_pks) == 0:
        return []
    cand_pks = cand_pks[np.argsort(env[cand_pks])[::-1]]

    lookback_n = int(round(lookback_s * fs))
    tail_n = max(3, int(round(0.005 * fs)))
    guard = max(1, int(round(min_spacing_s * fs)))
    accepted_centers: list[int] = []
    arrivals: list[ChirpArrival] = []

    for pk in cand_pks:
        pk_i = int(pk)
        if any(abs(pk_i - prev) < guard for prev in accepted_centers):
            continue
        idx_rel, snr_db, ncc = _pick_first_arrival_with_ncc(
            env=env,
            x_filt=x_filt,
            tpl_len=tpl_len,
            fs=fs,
            start=max(0, pk_i - lookback_n),
            end=min(len(env), pk_i + tail_n),
            sigma=sigma,
            min_snr_db=min_snr_db,
            min_ncc=min_ncc,
            lookback_s=lookback_s,
            direct_path_ratio=direct_path_ratio,
            tpl_halves=tpl_halves,
        )
        if idx_rel is None:
            # Skip loud non-chirp claps/impulses instead of stopping early
            continue
        i_center = int(round(idx_rel))
        if any(abs(i_center - prev) < guard for prev in accepted_centers):
            continue
        accepted_centers.append(i_center)
        arrivals.append(
            ChirpArrival(
                sample_index=float(start_sample_index) + idx_rel,
                local_offset=idx_rel,
                snr_db=snr_db,
                ncc=ncc,
            )
        )

    arrivals.sort(key=lambda a: a.sample_index)
    return arrivals


def detect_chirp_arrivals(
    samples_f32: np.ndarray,
    fs: int,
    mode: CalibrationMode = "lfm_dual",
    gap_s: float = DUAL_CHIRP_GAP_S,
    min_snr_db: float = 12.0,
    min_ncc: float = DEFAULT_CHIRP_MIN_NCC,
    valid_mask: np.ndarray | None = None,
    lookback_s: float = 0.025,
    direct_path_ratio: float = 0.55,
    f_low: float = 1200.0,
    f_high: float = 8500.0,
) -> tuple[list[float], float]:
    """Detect fractional chirp start sample indices inside samples_f32."""
    prep = _compute_chirp_envelope(
        samples_f32, fs=fs, valid_mask=valid_mask, f_low=f_low, f_high=f_high,
    )
    if prep is None:
        return [], -100.0
    env, x_filt, tpl, sigma, tpl_len = prep
    tpl_halves = _precompute_template_halves(tpl)

    min_height = sigma * (10.0 ** (min_snr_db / 20.0))
    lookback_n = int(round(lookback_s * fs))
    tail_n = max(3, int(round(0.005 * fs)))
    cand_pks, _ = find_peaks(env, height=min_height, distance=max(1, int(round(0.05 * fs))))
    p_a: float | None = None
    snr_a = 20.0 * math.log10(max(float(np.max(env)), 1e-12) / max(sigma, 1e-12)) if len(env) else -100.0
    if len(cand_pks):
        cand_pks = cand_pks[np.argsort(env[cand_pks])[::-1]]
        for pk in cand_pks:
            pk_i = int(pk)
            p_cand, snr_cand = _pick_first_arrival(
                env, x_filt, tpl_len, fs,
                max(0, pk_i - lookback_n), min(len(env), pk_i + tail_n),
                sigma, min_snr_db, min_ncc,
                lookback_s=lookback_s, direct_path_ratio=direct_path_ratio, tpl_halves=tpl_halves,
            )
            if p_cand is not None:
                p_a, snr_a = p_cand, snr_cand
                break
    if p_a is None:
        return [], snr_a

    if mode == "lfm_single":
        return [p_a], snr_a

    tol = int(round(0.08 * fs))
    gap_n = int(round(gap_s * fs))
    partner_min_snr = max(9.0, min_snr_db - 3.0)

    fwd_s = int(round(p_a)) + gap_n - tol
    fwd_e = int(round(p_a)) + gap_n + tol
    p_fwd, snr_fwd = _pick_first_arrival(
        env, x_filt, tpl_len, fs, fwd_s, fwd_e, sigma, partner_min_snr, min_ncc,
        lookback_s=lookback_s, direct_path_ratio=direct_path_ratio, tpl_halves=tpl_halves,
    )

    bwd_s = int(round(p_a)) - gap_n - tol
    bwd_e = int(round(p_a)) - gap_n + tol
    p_bwd, snr_bwd = _pick_first_arrival(
        env, x_filt, tpl_len, fs, bwd_s, bwd_e, sigma, partner_min_snr, min_ncc,
        lookback_s=lookback_s, direct_path_ratio=direct_path_ratio, tpl_halves=tpl_halves,
    )

    candidates: list[tuple[float, float]] = []
    if p_fwd is not None:
        candidates.append((p_fwd, snr_fwd))
    if p_bwd is not None:
        candidates.append((p_bwd, snr_bwd))

    if not candidates:
        return [p_a], snr_a

    p_b, snr_b = max(candidates, key=lambda item: item[1])
    arrivals = sorted([p_a, p_b])
    return arrivals, min(snr_a, snr_b)


@dataclass
class _ClockModel:
    a_s: float  # global time intercept at sample 0
    b_s: float  # seconds per local sample (1 / effective_fs)
    fs_nom: int
    drift_ppm: float
    chirp_snr_db: float
    residual_ms: float = 0.0
    last_sync_ts: float = 0.0
    synced: bool = True
    cal_points: list[tuple[float, float]] = field(default_factory=list)  # (n_abs, T_emit + tau_prop)


class SyncEngine:
    """Acoustic STO + SRO (clock drift) calibration and sub-sample timeline aligner."""

    def __init__(
        self,
        *,
        chirp_lookback_s: float = 0.025,
        chirp_direct_path_ratio: float = 0.55,
        chirp_f_low: float = 1200.0,
        chirp_f_high: float = 8500.0,
        chirp_min_ncc: float = DEFAULT_CHIRP_MIN_NCC,
    ) -> None:
        self.chirp_lookback_s = chirp_lookback_s
        self.chirp_direct_path_ratio = chirp_direct_path_ratio
        self.chirp_f_low = chirp_f_low
        self.chirp_f_high = chirp_f_high
        self.chirp_min_ncc = chirp_min_ncc
        self._models: dict[str, _ClockModel] = {}
        self._v2_fitter = ReferenceClockFitter()

    # ------------------------------------------------------------------
    # v2 Facade (CONTRACTS.md §2 & §7)
    # ------------------------------------------------------------------
    @property
    def ref_phone_id(self) -> str | None:
        return self._v2_fitter.ref_phone_id

    @property
    def needs_recalibration(self) -> bool:
        return self._v2_fitter.needs_recalibration

    @property
    def recalibration_reason(self) -> str | None:
        return self._v2_fitter.recalibration_reason

    def clock(self, phone_id: str) -> ClockModel | None:
        return self._v2_fitter.clock(phone_id)

    def to_global(self, phone_id: str, sample_index: float) -> float | None:
        t_v2 = self._v2_fitter.to_global(phone_id, sample_index)
        if t_v2 is not None:
            return t_v2
        model = self._models.get(phone_id)
        if model is not None and model.synced:
            return model.a_s + model.b_s * float(sample_index)
        return None

    def to_local(self, phone_id: str, t_global: float) -> float | None:
        n_v2 = self._v2_fitter.to_local(phone_id, t_global)
        if n_v2 is not None:
            return n_v2
        model = self._models.get(phone_id)
        if model is not None and model.synced and abs(model.b_s) > 1e-18:
            return (float(t_global) - model.a_s) / model.b_s
        return None

    def load_fitter(self, fitter: ReferenceClockFitter) -> None:
        """Install an existing fitted ReferenceClockFitter (e.g. from offline replay)."""
        self._v2_fitter = fitter
        self._models.clear()

    def remove_phone(self, phone_id: str) -> None:
        """Remove a disconnected phone and re-reference if it was the global clock reference."""
        self._models.pop(phone_id, None)
        self._v2_fitter.invalidate_phone(phone_id)

    def add_chirp_observation(
        self,
        phone_buffers: dict[str, PhoneAudioSlice],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        mode: CalibrationMode = "lfm_single",
    ) -> CalibrationResult:
        """Detect chirps in slices, pair across phones, and fit/update reference clock models."""
        requires_dual = mode == "lfm_dual" or mode == "ggwave_hybrid"

        # If the current reference phone dropped/disconnected, re-reference before adding new chirps
        cur_ref = self._v2_fitter.ref_phone_id
        if cur_ref is not None and (cur_ref not in phones or not phones[cur_ref].connected):
            self._v2_fitter.invalidate_phone(cur_ref, reason="Reference phone disconnected")

        arrivals_by_phone: dict[str, list[ChirpArrival]] = {}
        per_phone: dict[str, PhoneSyncReport] = {}
        sample_rates: dict[str, int] = {}
        slice_ends: dict[str, int] = {}
        latest_snr_by_phone: dict[str, float] = {}

        for pid, pcfg in phones.items():
            slc = phone_buffers.get(pid)
            if slc is None or len(slc.samples_f32) == 0:
                self._v2_fitter.invalidate_phone(pid)
                per_phone[pid] = PhoneSyncReport(synced=False, error="No audio slice available; please recalibrate")
                continue

            valid_mask = getattr(slc, "valid_mask", None)
            if requires_dual:
                rel_arrivals, snr_db = detect_chirp_arrivals(
                    slc.samples_f32,
                    fs=slc.sample_rate,
                    mode=mode,
                    min_ncc=self.chirp_min_ncc,
                    valid_mask=valid_mask,
                    lookback_s=self.chirp_lookback_s,
                    direct_path_ratio=self.chirp_direct_path_ratio,
                    f_low=self.chirp_f_low,
                    f_high=self.chirp_f_high,
                )
                arrivals = []
                if rel_arrivals:
                    prep = _compute_chirp_envelope(
                        slc.samples_f32,
                        fs=slc.sample_rate,
                        valid_mask=valid_mask,
                        f_low=self.chirp_f_low,
                        f_high=self.chirp_f_high,
                    )
                    if prep is not None:
                        env, x_filt, _, _, tpl_len = prep
                        for idx_rel in rel_arrivals:
                            if valid_mask is not None:
                                lo = max(0, int(math.floor(idx_rel)))
                                hi = min(len(valid_mask), int(math.ceil(idx_rel)) + tpl_len)
                                if not bool(np.all(valid_mask[lo:hi])):
                                    continue
                            i_c = min(len(env) - 1, max(0, int(round(idx_rel))))
                            arrivals.append(
                                ChirpArrival(
                                    sample_index=float(slc.start_sample_index) + idx_rel,
                                    local_offset=idx_rel,
                                    snr_db=snr_db,
                                    ncc=_local_template_ncc(env, x_filt, i_c, tpl_len),
                                )
                            )
            else:
                arrivals = detect_all_chirps(
                    slc.samples_f32,
                    fs=slc.sample_rate,
                    start_sample_index=slc.start_sample_index,
                    valid_mask=valid_mask,
                    min_ncc=self.chirp_min_ncc,
                    lookback_s=self.chirp_lookback_s,
                    direct_path_ratio=self.chirp_direct_path_ratio,
                    f_low=self.chirp_f_low,
                    f_high=self.chirp_f_high,
                )
                snr_db = max((a.snr_db for a in arrivals), default=-100.0)

            if not arrivals:
                self._v2_fitter.invalidate_phone(pid)
                per_phone[pid] = PhoneSyncReport(
                    synced=False,
                    chirp_snr_db=round(snr_db, 2),
                    error=f"Chirp not detected (SNR={snr_db:.1f} dB); please recalibrate",
                )
                continue

            if requires_dual and len(arrivals) < 2:
                self._v2_fitter.invalidate_phone(pid)
                per_phone[pid] = PhoneSyncReport(
                    synced=False,
                    chirp_snr_db=round(snr_db, 2),
                    error=(
                        f"Only {len(arrivals)} of 2 calibration chirps detected (SNR={snr_db:.1f} dB); "
                        "please recalibrate"
                    ),
                )
                continue

            arrivals_by_phone[pid] = arrivals
            sample_rates[pid] = slc.sample_rate
            slice_ends[pid] = slc.start_sample_index + len(slc.samples_f32)
            latest_snr_by_phone[pid] = snr_db

        # Ensure sample_rates covers already-stored phones too
        for em in self._v2_fitter.emissions:
            for pid, obs in em.obs_by_phone.items():
                sample_rates.setdefault(pid, obs.sample_rate)

        self._v2_fitter.ingest_arrivals(
            arrivals_by_phone=arrivals_by_phone,
            sample_rates=sample_rates,
            slice_end_indices=slice_ends,
            phones=phones,
            room=room,
        )

        # Pairwise GCC-PHAT cross-verification for phones with < 3 points (honest 1- or 2-chirp residual)
        synced_pids = [pid for pid in phones if self._v2_fitter.clock(pid) is not None]
        pairwise_res_ms: dict[str, float] = {}
        if len(synced_pids) >= 2 and self._v2_fitter.emissions:
            # Verify on the latest emission containing >= 2 synced phones
            target_em = None
            for em in reversed(self._v2_fitter.emissions):
                if sum(1 for pid in synced_pids if pid in em.obs_by_phone and pid in phone_buffers) >= 2:
                    target_em = em
                    break
            if target_em is not None:
                em_times: list[float] = []
                tau_props: dict[str, float] = {}
                for pid in synced_pids:
                    if pid in target_em.obs_by_phone and pid in phone_buffers:
                        tau_p = flight_time_s(phones[pid], room)
                        tau_props[pid] = tau_p
                        t_g = self.to_global(pid, target_em.obs_by_phone[pid].sample_index)
                        if t_g is not None:
                            em_times.append(t_g - tau_p)
                if em_times:
                    t_emit = float(np.median(em_times))
                    win_s = t_emit - 0.015
                    win_e = t_emit + max(tau_props.values()) + CHIRP_DURATION_S + 0.025
                    common = self.get_common_time_range({pid: phone_buffers[pid] for pid in tau_props})
                    if common is not None:
                        win_s = max(win_s, common[0] + 0.001)
                        win_e = min(win_e, common[1] - 0.001)
                    if win_e - win_s > CHIRP_DURATION_S:
                        aligned_cal = self.extract_aligned_window(
                            {pid: phone_buffers[pid] for pid in tau_props}, win_s, win_e, target_fs=48000
                        )
                        pair_errs: dict[str, list[float]] = {pid: [] for pid in aligned_cal}
                        cal_pids = list(aligned_cal.keys())
                        for ia in range(len(cal_pids)):
                            for ib in range(ia + 1, len(cal_pids)):
                                pa, pb = cal_pids[ia], cal_pids[ib]
                                expected_tdoa = tau_props[pa] - tau_props[pb]
                                lags, corr, measured_tdoa, _ = compute_gcc_phat(
                                    aligned_cal[pa],
                                    aligned_cal[pb],
                                    fs=48000,
                                    max_tau_s=abs(expected_tdoa) + 0.015,
                                    f_min=1200.0,
                                    f_max=8000.0,
                                )
                                if abs(measured_tdoa - expected_tdoa) > 0.002 and len(lags) > 0:
                                    # Prefer direct-path GCC peak near expected TDOA when a room bounce slightly wins
                                    win_mask = np.abs(lags - expected_tdoa) <= 0.0025
                                    g_peak = float(np.max(corr))
                                    if g_peak > 1e-12 and bool(np.any(win_mask)):
                                        sub_corr = corr[win_mask]
                                        k_loc = int(np.argmax(sub_corr))
                                        if float(sub_corr[k_loc]) >= 0.50 * g_peak:
                                            measured_tdoa = float(lags[win_mask][k_loc])
                                err_ms = abs(measured_tdoa - expected_tdoa) * 1000.0
                                pair_errs[pa].append(err_ms)
                                pair_errs[pb].append(err_ms)
                        for pid, errs in pair_errs.items():
                            if errs:
                                pairwise_res_ms[pid] = float(np.median(errs))

        residuals_ms: list[float] = []
        for pid in phones:
            model = self._v2_fitter.clock(pid)
            if model is None:
                if pid not in per_phone:
                    per_phone[pid] = PhoneSyncReport(
                        synced=False,
                        error=self._v2_fitter.recalibration_reason or "Could not align phone; please recalibrate",
                    )
                continue
            res_ms = model.residual_ms if model.n_points >= 3 else pairwise_res_ms.get(pid, model.residual_ms)
            model.residual_ms = round(res_ms, 4)
            residuals_ms.append(res_ms)
            per_phone[pid] = PhoneSyncReport(
                synced=True,
                residual_ms=round(res_ms, 4),
                drift_ppm=round(model.drift_ppm, 2),
                chirp_snr_db=round(latest_snr_by_phone.get(pid, 0.0), 2) if pid in latest_snr_by_phone else None,
            )

        return CalibrationResult(
            mode=mode,
            success=len(synced_pids) >= 2 and not self._v2_fitter.needs_recalibration,
            synced_count=len(synced_pids),
            median_residual_ms=round(float(np.median(residuals_ms)), 4) if residuals_ms else None,
            max_residual_ms=round(float(np.max(residuals_ms)), 4) if residuals_ms else None,
            per_phone=per_phone,
        )

    # ------------------------------------------------------------------
    # Legacy + Shared Methods
    # ------------------------------------------------------------------
    def is_synced(self, phone_id: str) -> bool:
        if self._v2_fitter.clock(phone_id) is not None:
            return True
        model = self._models.get(phone_id)
        return bool(model is not None and model.synced)

    def mark_clock_jump(self, phone_id: str) -> None:
        model = self._models.get(phone_id)
        if model is not None:
            model.synced = False
            model.cal_points.clear()
        self._v2_fitter.invalidate_phone(phone_id)

    def sample_to_global_time(self, phone_id: str, sample_index: float) -> float:
        t_glob = self.to_global(phone_id, sample_index)
        if t_glob is None:
            raise KeyError(f"Phone '{phone_id}' is not synced")
        return t_glob

    def get_common_time_range(
        self,
        phone_buffers: dict[str, PhoneAudioSlice],
    ) -> tuple[float, float] | None:
        """Return overlapping [t_start, t_end] global seconds across all synced phones in buffers."""
        starts: list[float] = []
        ends: list[float] = []
        for pid, slc in phone_buffers.items():
            if not self.is_synced(pid) or len(slc.samples_f32) == 0:
                continue
            t0 = self.to_global(pid, slc.start_sample_index)
            t1 = self.to_global(pid, slc.start_sample_index + len(slc.samples_f32) - 1)
            if t0 is None or t1 is None:
                continue
            starts.append(t0)
            ends.append(t1)
        if len(starts) < 2:
            return None
        lo = max(starts)
        hi = min(ends)
        return (lo, hi) if hi > lo else None

    def extract_aligned_window(
        self,
        phone_buffers: dict[str, PhoneAudioSlice],
        t_start_s: float,
        t_end_s: float,
        target_fs: int = 48000,
        exclude_clipped: bool = False,
        clip_threshold: float = CLIP_THRESHOLD,
    ) -> dict[str, np.ndarray]:
        """Resample all eligible synced phones onto a common uniform global timeline at target_fs.

        Gap-aware: excludes any phone whose captured samples do not cover [t_start_s, t_end_s],
        whose valid_mask contains False anywhere inside the window, or (when exclude_clipped=True)
        whose raw samples clip inside the window, while preserving its clock model.
        """
        if t_end_s <= t_start_s:
            return {}
        n_out = int(round((t_end_s - t_start_s) * target_fs))
        if n_out < 2:
            return {}

        t_grid = t_start_s + np.arange(n_out, dtype=np.float64) / float(target_fs)
        aligned: dict[str, np.ndarray] = {}
        for pid, slc in phone_buffers.items():
            if not self.is_synced(pid) or len(slc.samples_f32) < 4:
                continue
            idx_start_abs = self.to_local(pid, float(t_grid[0]))
            idx_end_abs = self.to_local(pid, float(t_grid[-1]))
            if idx_start_abs is None or idx_end_abs is None:
                continue

            lo_rel = idx_start_abs - float(slc.start_sample_index)
            hi_rel = idx_end_abs - float(slc.start_sample_index)
            n_samples = len(slc.samples_f32)
            # Exclude phone if requested window falls outside captured samples (keep clock model!)
            if lo_rel < 0.0 or hi_rel > float(n_samples - 1) or hi_rel <= lo_rel:
                continue

            i_lo = max(0, int(math.floor(lo_rel)))
            i_hi = min(n_samples - 1, int(math.ceil(hi_rel)))
            valid_mask = getattr(slc, "valid_mask", None)
            if valid_mask is not None:
                if len(valid_mask) != n_samples:
                    continue
                if not bool(np.all(valid_mask[i_lo : i_hi + 1])):
                    continue

            if exclude_clipped and is_window_clipped(slc.samples_f32[i_lo : i_hi + 1], clip_threshold=clip_threshold):
                continue

            n_local = np.linspace(lo_rel, hi_rel, n_out, dtype=np.float64)
            aligned[pid] = map_coordinates(
                slc.samples_f32.astype(np.float64),
                [n_local],
                order=3,
                mode="nearest",
            ).astype(np.float32)
        return aligned

    def calibrate_from_buffers(
        self,
        phone_buffers: dict[str, PhoneAudioSlice],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        mode: CalibrationMode = "lfm_dual",
    ) -> CalibrationResult:
        """Calibrate clock offsets (STO) and sample-rate drift (SRO) using acoustic chirps."""
        c = speed_of_sound(room.temperature_c)
        now_ts = time.time()

        detected: dict[str, tuple[list[float], float, float, PhoneAudioSlice]] = {}
        per_phone: dict[str, PhoneSyncReport] = {}
        requires_dual = mode in ("lfm_dual", "ggwave_hybrid")

        for pid, pcfg in phones.items():
            slc = phone_buffers.get(pid)
            if slc is None or len(slc.samples_f32) == 0:
                self.mark_clock_jump(pid)
                per_phone[pid] = PhoneSyncReport(synced=False, error="No audio slice available; please recalibrate")
                continue

            arrivals, snr_db = detect_chirp_arrivals(
                slc.samples_f32,
                fs=slc.sample_rate,
                mode=mode,
                min_ncc=self.chirp_min_ncc,
                valid_mask=slc.valid_mask,
                lookback_s=self.chirp_lookback_s,
                direct_path_ratio=self.chirp_direct_path_ratio,
                f_low=self.chirp_f_low,
                f_high=self.chirp_f_high,
            )
            if not arrivals:
                self.mark_clock_jump(pid)
                per_phone[pid] = PhoneSyncReport(
                    synced=False,
                    chirp_snr_db=round(snr_db, 2),
                    error=f"Chirp not detected (SNR={snr_db:.1f} dB); please recalibrate",
                )
                continue

            if requires_dual and len(arrivals) < 2:
                self.mark_clock_jump(pid)
                per_phone[pid] = PhoneSyncReport(
                    synced=False,
                    chirp_snr_db=round(snr_db, 2),
                    error=(
                        f"Only {len(arrivals)} of 2 calibration chirps detected (SNR={snr_db:.1f} dB); "
                        "please recalibrate"
                    ),
                )
                continue

            tau_prop = math.hypot(pcfg.x_m - room.speaker_x_m, pcfg.y_m - room.speaker_y_m) / c
            abs_arrivals = [float(slc.start_sample_index) + n_rel for n_rel in arrivals]
            detected[pid] = (abs_arrivals, tau_prop, snr_db, slc)

        # Keep continuous global clock epoch across re-calibrations if previously synced phones exist
        anchor_t0 = 0.0
        prior_t0s: list[float] = []
        for pid, (abs_arrivals, tau_prop, _, _) in detected.items():
            prev = self._models.get(pid)
            if prev is not None and prev.synced and prev.cal_points:
                prior_t0s.append(prev.a_s + prev.b_s * abs_arrivals[0] - tau_prop)
        if prior_t0s:
            anchor_t0 = float(np.median(prior_t0s))

        # Fit linear clock model per phone
        for pid, (abs_arrivals, tau_prop, snr_db, slc) in detected.items():
            fs_nom = slc.sample_rate
            prev = self._models.get(pid)
            points = list(prev.cal_points) if (prev is not None and prev.synced) else []

            if len(abs_arrivals) >= 2:
                n0, n1 = abs_arrivals[0], abs_arrivals[1]
                points.append((n0, anchor_t0 + tau_prop))
                points.append((n1, anchor_t0 + DUAL_CHIRP_GAP_S + tau_prop))
            else:
                points.append((abs_arrivals[0], anchor_t0 + tau_prop))

            if len(points) >= 2:
                ns = np.array([pt[0] for pt in points], dtype=np.float64)
                ts = np.array([pt[1] for pt in points], dtype=np.float64)
                span_t = float(np.max(ts) - np.min(ts))
                if span_t > 0.5:
                    n_mean = float(np.mean(ns))
                    t_mean = float(np.mean(ts))
                    dn = ns - n_mean
                    denom = float(np.dot(dn, dn))
                    b_s = float(np.dot(dn, ts - t_mean) / denom) if denom > 0.0 else (1.0 / fs_nom)
                    f_hat = 1.0 / b_s if b_s > 0.0 else float(fs_nom)
                    drift_ppm = ((f_hat - fs_nom) / float(fs_nom)) * 1e6
                    if abs(drift_ppm) > 500.0:
                        b_s = 1.0 / float(fs_nom)
                        drift_ppm = 0.0
                    a_s = t_mean - b_s * n_mean
                else:
                    b_s = 1.0 / float(fs_nom)
                    a_s = ts[-1] - b_s * ns[-1]
                    drift_ppm = 0.0
            else:
                b_s = 1.0 / float(fs_nom)
                a_s = points[0][1] - b_s * points[0][0]
                drift_ppm = 0.0

            self._models[pid] = _ClockModel(
                a_s=a_s,
                b_s=b_s,
                fs_nom=fs_nom,
                drift_ppm=drift_ppm,
                chirp_snr_db=snr_db,
                last_sync_ts=now_ts,
                synced=True,
                cal_points=points[-12:],
            )

        # Honest pairwise cross-verification using aligned Chirp 1 waveforms
        synced_pids = list(detected.keys())
        residuals_ms: dict[str, float] = {pid: 0.0 for pid in synced_pids}
        if len(synced_pids) >= 2:
            max_tau_prop = max(detected[pid][1] for pid in synced_pids)
            cal_slices = {pid: detected[pid][3] for pid in synced_pids}
            win_start = anchor_t0 - 0.015
            win_end = anchor_t0 + max_tau_prop + CHIRP_DURATION_S + 0.025
            common = self.get_common_time_range(cal_slices)
            if common is not None:
                win_start = max(win_start, common[0] + 0.001)
                win_end = min(win_end, common[1] - 0.001)
            aligned_cal = self.extract_aligned_window(cal_slices, win_start, win_end, target_fs=48000)

            pair_errs: dict[str, list[float]] = {pid: [] for pid in synced_pids}
            for idx_a in range(len(synced_pids)):
                for idx_b in range(idx_a + 1, len(synced_pids)):
                    pa, pb = synced_pids[idx_a], synced_pids[idx_b]
                    if pa not in aligned_cal or pb not in aligned_cal:
                        continue
                    expected_tdoa = detected[pa][1] - detected[pb][1]
                    max_tau = abs(expected_tdoa) + 0.015
                    _, _, measured_tdoa, _ = compute_gcc_phat(
                        aligned_cal[pa],
                        aligned_cal[pb],
                        fs=48000,
                        max_tau_s=max_tau,
                        f_min=1200.0,
                        f_max=8000.0,
                    )
                    err_ms = abs(measured_tdoa - expected_tdoa) * 1000.0
                    pair_errs[pa].append(err_ms)
                    pair_errs[pb].append(err_ms)

            for pid in synced_pids:
                residuals_ms[pid] = float(np.median(pair_errs[pid])) if pair_errs[pid] else 0.0

        for pid in synced_pids:
            model = self._models[pid]
            model.residual_ms = residuals_ms[pid]
            per_phone[pid] = PhoneSyncReport(
                synced=True,
                residual_ms=round(model.residual_ms, 4),
                drift_ppm=round(model.drift_ppm, 2),
                chirp_snr_db=round(model.chirp_snr_db, 2),
            )

        all_res = [residuals_ms[pid] for pid in synced_pids]
        return CalibrationResult(
            mode=mode,
            success=len(synced_pids) >= 2,
            synced_count=len(synced_pids),
            median_residual_ms=round(float(np.median(all_res)), 4) if all_res else None,
            max_residual_ms=round(float(np.max(all_res)), 4) if all_res else None,
            per_phone=per_phone,
        )
