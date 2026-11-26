"""v2 reference-phone clock fitter, repeated chirp pairing, and held-out evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math

import numpy as np

from shared import protocol as _protocol
from shared.protocol import PhoneAudioSlice, PhoneConfig, RoomConfig, speed_of_sound

CLOCK_FIT_MAX_AGE_S: float = float(getattr(_protocol, "CLOCK_FIT_MAX_AGE_S", 600.0))
_SLICE_HAS_VALID_MASK: bool = "valid_mask" in inspect.signature(PhoneAudioSlice).parameters


@dataclass(slots=True)
class AudioSliceV2:
    """DSP-local PhoneAudioSlice extension carrying valid_mask before shared/protocol.py v2 lands."""

    phone_id: str
    samples_f32: np.ndarray
    start_sample_index: int
    sample_rate: int
    valid_mask: np.ndarray | None = None


def make_phone_slice(
    phone_id: str,
    samples_f32: np.ndarray,
    start_sample_index: int,
    sample_rate: int,
    valid_mask: np.ndarray | None = None,
) -> PhoneAudioSlice | AudioSliceV2:
    """Construct PhoneAudioSlice with valid_mask if supported by shared.protocol, else AudioSliceV2."""
    if _SLICE_HAS_VALID_MASK:
        return PhoneAudioSlice(  # type: ignore[call-arg]
            phone_id=phone_id,
            samples_f32=samples_f32,
            start_sample_index=start_sample_index,
            sample_rate=sample_rate,
            valid_mask=valid_mask,
        )
    if valid_mask is None:
        return PhoneAudioSlice(
            phone_id=phone_id,
            samples_f32=samples_f32,
            start_sample_index=start_sample_index,
            sample_rate=sample_rate,
        )
    return AudioSliceV2(
        phone_id=phone_id,
        samples_f32=samples_f32,
        start_sample_index=start_sample_index,
        sample_rate=sample_rate,
        valid_mask=valid_mask,
    )

if hasattr(_protocol, "ClockModel"):
    ClockModel = _protocol.ClockModel
else:

    @dataclass(slots=True)
    class ClockModel:  # type: ignore[no-redef]
        """Maps a phone's local sample index to the global timeline."""

        phone_id: str
        ref_phone_id: str
        alpha: float
        beta: float
        n_points: int
        residual_ms: float
        drift_ppm: float


@dataclass(slots=True)
class ChirpArrival:
    """Single detected chirp arrival inside a phone's audio slice."""

    sample_index: float  # absolute AudioWorklet sample index
    local_offset: float  # sample index relative to slice start
    snr_db: float
    ncc: float


@dataclass(slots=True)
class PhoneObservation:
    """Flight-time-corrected chirp observation for one phone at one physical emission."""

    sample_index: float
    flight_corrected_index: float  # m_ik = n_ik - fs_i * tau_prop_i
    sample_rate: int
    snr_db: float
    ncc: float


@dataclass(slots=True)
class EmissionRecord:
    """One physical chirp emission observed across one or more phones."""

    emission_id: int
    obs_by_phone: dict[str, PhoneObservation] = field(default_factory=dict)


@dataclass(slots=True)
class HeldOutEvaluation:
    """Result of fitting on early chirps and scoring a held-out chirp before fitting it."""

    fit_chirp_indices: list[int]
    held_out_chirp_index: int
    ref_phone_id: str | None
    fit_residuals_ms: dict[str, float]
    phones_expected: list[str]
    phones_measured: list[str]
    phones_missing: list[str]
    coverage_complete: bool
    emission_times_s: dict[str, float]
    per_phone_diff_ms: dict[str, float]
    worst_pairwise_ms: float | None
    gate_passed: bool
    status: str  # "pass" | "fail" | "incomplete_coverage"


def flight_time_s(phone: PhoneConfig, room: RoomConfig) -> float:
    """Direct-path acoustic flight time from calibration speaker to phone in seconds."""
    c = speed_of_sound(room.temperature_c)
    return math.hypot(phone.x_m - room.speaker_x_m, phone.y_m - room.speaker_y_m) / c


def pair_arrivals_to_emissions(
    arrivals_by_phone: dict[str, list[ChirpArrival]],
    sample_rates: dict[str, int],
    slice_end_indices: dict[str, int] | None,
    phones: dict[str, PhoneConfig],
    room: RoomConfig,
    tol_s: float = 0.06,
    start_emission_id: int = 0,
) -> list[EmissionRecord]:
    """Pair chirp arrivals from the same physical emission across phones."""
    non_empty = {pid: sorted(arrs, key=lambda a: a.sample_index) for pid, arrs in arrivals_by_phone.items() if arrs}
    if not non_empty:
        return []

    obs_by_pid: dict[str, list[PhoneObservation]] = {}
    for pid, arrs in non_empty.items():
        fs_i = sample_rates[pid]
        tau_i = flight_time_s(phones[pid], room)
        obs_by_pid[pid] = [
            PhoneObservation(
                sample_index=a.sample_index,
                flight_corrected_index=a.sample_index - fs_i * tau_i,
                sample_rate=fs_i,
                snr_db=a.snr_db,
                ncc=a.ncc,
            )
            for a in arrs
        ]

    # Anchor on the phone with the most detected chirps (tie-break: highest mean SNR)
    anchor_pid = max(
        obs_by_pid.keys(),
        key=lambda pid: (len(obs_by_pid[pid]), float(np.mean([o.snr_db for o in obs_by_pid[pid]]))),
    )
    fs_anchor = sample_rates[anchor_pid]
    t_anchor = [o.flight_corrected_index / fs_anchor for o in obs_by_pid[anchor_pid]]
    anchor_arr0 = non_empty[anchor_pid][0]
    anchor_ref_s = (
        slice_end_indices[anchor_pid] / fs_anchor
        if slice_end_indices is not None and anchor_pid in slice_end_indices
        else (anchor_arr0.sample_index - anchor_arr0.local_offset) / fs_anchor
    )

    groups: list[dict[str, PhoneObservation]] = [{anchor_pid: obs} for obs in obs_by_pid[anchor_pid]]

    for pid, obs_list in obs_by_pid.items():
        if pid == anchor_pid:
            continue
        fs_i = sample_rates[pid]
        t_i = [o.flight_corrected_index / fs_i for o in obs_list]
        pid_arr0 = non_empty[pid][0]
        pid_ref_s = (
            slice_end_indices[pid] / fs_i
            if slice_end_indices is not None and pid in slice_end_indices
            else (pid_arr0.sample_index - pid_arr0.local_offset) / fs_i
        )
        delta_hint = anchor_ref_s - pid_ref_s

        # Candidate shifts delta such that t_i + delta aligns with t_anchor
        best_score: tuple[int, float, float] | None = None
        best_pairs: list[tuple[int, int]] = []
        for idx_a, ta in enumerate(t_anchor):
            for idx_b, tb in enumerate(t_i):
                delta = ta - tb
                used_a: set[int] = set()
                pairs: list[tuple[int, int]] = []
                err_sum = 0.0
                for j, tj in enumerate(t_i):
                    t_shifted = tj + delta
                    best_k = None
                    best_err = tol_s
                    for k, tk in enumerate(t_anchor):
                        if k in used_a:
                            continue
                        err = abs(t_shifted - tk)
                        if err <= best_err:
                            best_err = err
                            best_k = k
                    if best_k is not None:
                        used_a.add(best_k)
                        pairs.append((best_k, j))
                        err_sum += best_err
                # Prefer shifts consistent with coarse buffer alignment (within 2.0 s) before
                # sub-millisecond drift tie-breaking, preventing 5s/30s chirp-index misalignment.
                coarse_ok = 1 if abs(delta - delta_hint) <= 2.0 else 0
                score = (len(pairs), coarse_ok, -err_sum if coarse_ok else -abs(delta - delta_hint), -abs(delta - delta_hint))
                if best_score is None or score > best_score:
                    best_score = score
                    best_pairs = pairs

        matched_b = {j for _, j in best_pairs}
        for k_anchor, j_b in best_pairs:
            groups[k_anchor][pid] = obs_list[j_b]

        # Any unmatched arrival on pid becomes its own emission (sorted later)
        for j, obs in enumerate(obs_list):
            if j not in matched_b:
                groups.append({pid: obs})

    # Order emissions chronologically by anchor time (or shifted fallback)
    def _group_sort_key(grp: dict[str, PhoneObservation]) -> float:
        if anchor_pid in grp:
            return grp[anchor_pid].flight_corrected_index / fs_anchor
        pid0, obs0 = next(iter(grp.items()))
        return obs0.flight_corrected_index / sample_rates[pid0]

    groups.sort(key=_group_sort_key)
    return [EmissionRecord(emission_id=start_emission_id + idx, obs_by_phone=grp) for idx, grp in enumerate(groups)]


def _ols_clock_fit(
    m_i: np.ndarray,
    m_ref: np.ndarray,
    fs_i: int,
    fs_ref: int,
) -> tuple[float, float, float, float, np.ndarray]:
    """Return (alpha, beta, drift_ppm, residual_ms, errs_samples) for >= 2 paired chirp points."""
    span_s = float(np.max(m_ref) - np.min(m_ref)) / float(fs_ref)
    if span_s > 0.5:
        mi_mean = float(np.mean(m_i))
        mr_mean = float(np.mean(m_ref))
        dm = m_i - mi_mean
        denom = float(np.dot(dm, dm))
        beta = float(np.dot(dm, m_ref - mr_mean) / denom) if denom > 0.0 else (float(fs_ref) / float(fs_i))
        drift_ppm = (beta * float(fs_i) / float(fs_ref) - 1.0) * 1e6
        if abs(drift_ppm) > 500.0:
            beta = float(fs_ref) / float(fs_i)
            drift_ppm = 0.0
        alpha = float(mr_mean - beta * mi_mean)
    else:
        beta = float(fs_ref) / float(fs_i)
        alpha = float(m_ref[-1] - beta * m_i[-1])
        drift_ppm = 0.0
    errs = (alpha + beta * m_i) - m_ref
    residual_ms = float(np.sqrt(np.mean(errs ** 2)) / float(fs_ref) * 1000.0)
    return alpha, beta, drift_ppm, residual_ms, errs


class ReferenceClockFitter:
    """Maintains reference-phone global clock and per-phone OLS linear fits."""

    def __init__(self, max_age_s: float = CLOCK_FIT_MAX_AGE_S) -> None:
        self.max_age_s = max_age_s
        self.ref_phone_id: str | None = None
        self.ref_sample_rate: int = 48000
        self.emissions: list[EmissionRecord] = []
        self.models: dict[str, ClockModel] = {}
        self.needs_recalibration: bool = False
        self.recalibration_reason: str | None = None
        self._next_emission_id: int = 0

    def clock(self, phone_id: str) -> ClockModel | None:
        return self.models.get(phone_id)

    def to_global(self, phone_id: str, sample_index: float) -> float | None:
        model = self.models.get(phone_id)
        if model is None or self.ref_phone_id is None:
            return None
        return (model.alpha + model.beta * float(sample_index)) / float(self.ref_sample_rate)

    def to_local(self, phone_id: str, t_global: float) -> float | None:
        model = self.models.get(phone_id)
        if model is None or self.ref_phone_id is None or abs(model.beta) < 1e-12:
            return None
        return (float(t_global) * float(self.ref_sample_rate) - model.alpha) / model.beta

    def ingest_arrivals(
        self,
        arrivals_by_phone: dict[str, list[ChirpArrival]],
        sample_rates: dict[str, int],
        slice_end_indices: dict[str, int],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
    ) -> list[EmissionRecord]:
        """Deduplicate overlapping chirp detections, pair into emissions, and refit clocks."""
        deduped: dict[str, list[ChirpArrival]] = {}
        for pid, arrs in arrivals_by_phone.items():
            fs_i = sample_rates[pid]
            min_sep = 0.25 * fs_i
            seen_indices = [
                em.obs_by_phone[pid].sample_index for em in self.emissions if pid in em.obs_by_phone
            ]
            fresh: list[ChirpArrival] = []
            for arr in sorted(arrs, key=lambda a: a.sample_index):
                if any(abs(arr.sample_index - s) <= min_sep for s in seen_indices):
                    continue
                if any(abs(arr.sample_index - f.sample_index) <= min_sep for f in fresh):
                    continue
                fresh.append(arr)
            if fresh:
                deduped[pid] = fresh

        # Attach arrivals from newly catching-up phones to an existing synced emission when aligned
        remaining_for_pairing: dict[str, list[ChirpArrival]] = {}
        for pid, arrs in deduped.items():
            unmatched: list[ChirpArrival] = []
            fs_i = sample_rates[pid]
            tau_i = flight_time_s(phones[pid], room)
            for arr in arrs:
                attached = False
                if self.ref_phone_id is not None and len(arrs) == 1:
                    # Check if an already-synced peer in this batch maps this instant to an existing emission
                    for em in self.emissions:
                        if pid in em.obs_by_phone:
                            continue
                        for peer_pid, peer_arrs in arrivals_by_phone.items():
                            if peer_pid == pid or peer_pid not in em.obs_by_phone:
                                continue
                            peer_obs = em.obs_by_phone[peer_pid]
                            for pa in peer_arrs:
                                if abs(pa.sample_index - peer_obs.sample_index) <= 0.25 * peer_obs.sample_rate:
                                    t_peer_rel = (pa.sample_index / peer_obs.sample_rate) - flight_time_s(
                                        phones[peer_pid], room
                                    )
                                    t_i_rel = (arr.sample_index / fs_i) - tau_i
                                    if abs(t_peer_rel - t_i_rel) <= 0.10:
                                        em.obs_by_phone[pid] = PhoneObservation(
                                            sample_index=arr.sample_index,
                                            flight_corrected_index=arr.sample_index - fs_i * tau_i,
                                            sample_rate=fs_i,
                                            snr_db=arr.snr_db,
                                            ncc=arr.ncc,
                                        )
                                        attached = True
                                        break
                            if attached:
                                break
                        if attached:
                            break
                if not attached:
                    unmatched.append(arr)
            if unmatched:
                remaining_for_pairing[pid] = unmatched

        new_emissions = pair_arrivals_to_emissions(
            arrivals_by_phone=remaining_for_pairing,
            sample_rates=sample_rates,
            slice_end_indices=slice_end_indices,
            phones=phones,
            room=room,
            start_emission_id=self._next_emission_id,
        )
        self._next_emission_id += len(new_emissions)
        self.emissions.extend(new_emissions)
        self.refit(sample_rates)
        return new_emissions

    def ingest_emissions(
        self,
        emissions: list[EmissionRecord],
        sample_rates: dict[str, int],
    ) -> None:
        """Load pre-paired emissions (used for held-out scoring on early chirps only)."""
        self.emissions = [
            EmissionRecord(emission_id=em.emission_id, obs_by_phone=dict(em.obs_by_phone)) for em in emissions
        ]
        self._next_emission_id = max((em.emission_id for em in self.emissions), default=-1) + 1
        self.refit(sample_rates)

    def invalidate_phone(self, phone_id: str, reason: str | None = None) -> None:
        """Clear a phone's stale observations and clock model; re-reference if it was reference."""
        for em in self.emissions:
            em.obs_by_phone.pop(phone_id, None)
        self.emissions = [em for em in self.emissions if em.obs_by_phone]
        was_ref = phone_id == self.ref_phone_id
        self.models.pop(phone_id, None)

        if was_ref:
            self.ref_phone_id = None
            sample_rates = {
                pid: obs.sample_rate for em in self.emissions for pid, obs in em.obs_by_phone.items()
            }
            self.re_reference(sample_rates)
            if reason and not self.recalibration_reason:
                self.recalibration_reason = reason
        elif len(self.models) < 2 and self.ref_phone_id is not None:
            self.needs_recalibration = True
            if not self.recalibration_reason:
                self.recalibration_reason = reason or "Fewer than 2 synced phones remain; please recalibrate"

    def re_reference(self, sample_rates: dict[str, int]) -> None:
        """Select the best remaining phone that supports >= 2 synced phones and refit."""
        remaining_pids: dict[str, list[float]] = {}
        for em in self.emissions:
            for pid, obs in em.obs_by_phone.items():
                remaining_pids.setdefault(pid, []).append(obs.snr_db)

        if len(remaining_pids) < 2:
            self.ref_phone_id = None
            self.models.clear()
            self.needs_recalibration = True
            self.recalibration_reason = "Reference phone lost; fewer than 2 phones have observations (recalibrate)"
            return

        # Pick remaining phone with most shared multi-phone emissions, tie-broken by highest mean chirp SNR
        def _candidate_score(pid: str) -> tuple[int, int, float]:
            shared_peers: set[str] = set()
            shared_events = 0
            for em in self.emissions:
                if pid in em.obs_by_phone and len(em.obs_by_phone) >= 2:
                    shared_events += 1
                    shared_peers.update(k for k in em.obs_by_phone if k != pid)
            return (len(shared_peers), shared_events, float(np.mean(remaining_pids[pid])))

        best_ref = max(remaining_pids.keys(), key=_candidate_score)
        best_peers, _, _ = _candidate_score(best_ref)
        if best_peers < 1:
            self.ref_phone_id = None
            self.models.clear()
            self.needs_recalibration = True
            self.recalibration_reason = (
                "Reference phone lost and remaining observations share no common chirp; please recalibrate"
            )
            return

        self.ref_phone_id = best_ref
        self.ref_sample_rate = sample_rates.get(
            best_ref,
            next(em.obs_by_phone[best_ref].sample_rate for em in self.emissions if best_ref in em.obs_by_phone),
        )
        self.refit(sample_rates)

        # Surface recalibration notice if any remaining phone could not be bridged to new_ref
        orphaned = [pid for pid in remaining_pids if pid not in self.models]
        if len(self.models) < 2:
            self.models.clear()
            self.ref_phone_id = None
            self.needs_recalibration = True
            self.recalibration_reason = "Reference phone lost; insufficient shared chirps to bridge phones"
        elif orphaned:
            self.needs_recalibration = True
            self.recalibration_reason = (
                f"Reference changed to {best_ref}; phones {','.join(sorted(orphaned))} lack shared chirps (recalibrate)"
            )
        else:
            self.needs_recalibration = False
            self.recalibration_reason = None

    def refit(self, sample_rates: dict[str, int]) -> None:
        """Fit every phone onto the reference phone's sample clock using OLS over recent chirps."""
        if self.ref_phone_id is not None:
            has_ref = any(self.ref_phone_id in em.obs_by_phone for em in self.emissions)
            if not has_ref:
                self.ref_phone_id = None

        if self.ref_phone_id is None:
            # First successful calibration: choose phone with highest chirp SNR in earliest multi-phone emission
            for em in self.emissions:
                if len(em.obs_by_phone) >= 2:
                    ref_pid = max(em.obs_by_phone.keys(), key=lambda pid: em.obs_by_phone[pid].snr_db)
                    self.ref_phone_id = ref_pid
                    self.ref_sample_rate = em.obs_by_phone[ref_pid].sample_rate
                    break
            if self.ref_phone_id is None:
                self.models.clear()
                return

        ref_pid = self.ref_phone_id
        ref_obs_all = [em.obs_by_phone[ref_pid] for em in self.emissions if ref_pid in em.obs_by_phone]
        if not ref_obs_all:
            self.models.clear()
            return
        self.ref_sample_rate = sample_rates.get(ref_pid, ref_obs_all[-1].sample_rate)
        fs_ref = self.ref_sample_rate

        new_models: dict[str, ClockModel] = {
            ref_pid: ClockModel(
                phone_id=ref_pid,
                ref_phone_id=ref_pid,
                alpha=0.0,
                beta=1.0,
                n_points=len(ref_obs_all),
                residual_ms=0.0,
                drift_ppm=0.0,
            )
        }

        all_pids = {pid for em in self.emissions for pid in em.obs_by_phone}
        for pid in sorted(all_pids):
            if pid == ref_pid:
                continue
            shared = [
                (em.obs_by_phone[pid], em.obs_by_phone[ref_pid])
                for em in self.emissions
                if pid in em.obs_by_phone and ref_pid in em.obs_by_phone
            ]
            if not shared:
                continue

            latest_m_ref = max(r.flight_corrected_index for _, r in shared)
            shared = [
                (o, r)
                for o, r in shared
                if (latest_m_ref - r.flight_corrected_index) / float(fs_ref) <= self.max_age_s
            ]
            if not shared:
                continue

            fs_i = sample_rates.get(pid, shared[-1][0].sample_rate)
            m_i = np.array([o.flight_corrected_index for o, _ in shared], dtype=np.float64)
            m_ref = np.array([r.flight_corrected_index for _, r in shared], dtype=np.float64)
            n_pts = len(shared)

            if n_pts == 1:
                beta = float(fs_ref) / float(fs_i)
                alpha = float(m_ref[0] - beta * m_i[0])
                drift_ppm = 0.0
                residual_ms = 0.0
            else:
                alpha, beta, drift_ppm, residual_ms, errs_samples = _ols_clock_fit(m_i, m_ref, fs_i, fs_ref)
                # Guard against a single outlier chirp or pre-slip history poisoning the OLS line.
                if n_pts >= 3 and residual_ms > 0.6:
                    candidates: list[tuple[float, float, float, float]] = []
                    worst_idx = int(np.argmax(np.abs(errs_samples)))
                    loo_mask = np.arange(n_pts) != worst_idx
                    if float(np.max(m_ref[loo_mask]) - np.min(m_ref[loo_mask])) / float(fs_ref) > 0.5:
                        ca, cb, cd, cr, _ = _ols_clock_fit(m_i[loo_mask], m_ref[loo_mask], fs_i, fs_ref)
                        candidates.append((ca, cb, cd, cr))
                    for k in range(n_pts - 1, 1, -1):
                        sub_i, sub_ref = m_i[-k:], m_ref[-k:]
                        if float(np.max(sub_ref) - np.min(sub_ref)) / float(fs_ref) <= 0.5:
                            continue
                        ca, cb, cd, cr, _ = _ols_clock_fit(sub_i, sub_ref, fs_i, fs_ref)
                        if cr <= 0.4:
                            candidates.append((ca, cb, cd, cr))
                            break
                    if candidates:
                        best = min(candidates, key=lambda c: c[3])
                        if best[3] < 0.4 and best[3] < 0.4 * residual_ms:
                            alpha, beta, drift_ppm, residual_ms = best

            new_models[pid] = ClockModel(
                phone_id=pid,
                ref_phone_id=ref_pid,
                alpha=alpha,
                beta=beta,
                n_points=n_pts,
                residual_ms=round(residual_ms, 4),
                drift_ppm=round(drift_ppm, 3),
            )

        self.models = new_models
        if len(self.models) >= 2:
            self.needs_recalibration = False
            self.recalibration_reason = None


def evaluate_held_out_chirp(
    emissions: list[EmissionRecord],
    phones: dict[str, PhoneConfig],
    room: RoomConfig,
    sample_rates: dict[str, int],
    fit_chirps: int = 2,
    held_out_index: int = 2,
) -> HeldOutEvaluation:
    """Fit clocks using ONLY the first `fit_chirps` emissions and score `held_out_index` before fitting it."""
    phones_expected = sorted(phones.keys())
    if held_out_index < fit_chirps:
        raise ValueError(f"held_out_index ({held_out_index}) must be >= fit_chirps ({fit_chirps})")
    if len(emissions) <= held_out_index:
        return HeldOutEvaluation(
            fit_chirp_indices=list(range(min(fit_chirps, len(emissions)))),
            held_out_chirp_index=held_out_index,
            ref_phone_id=None,
            fit_residuals_ms={},
            phones_expected=phones_expected,
            phones_measured=[],
            phones_missing=phones_expected,
            coverage_complete=False,
            emission_times_s={},
            per_phone_diff_ms={},
            worst_pairwise_ms=None,
            gate_passed=False,
            status="incomplete_coverage",
        )

    fit_indices = list(range(fit_chirps))
    fitter = ReferenceClockFitter()
    fitter.ingest_emissions(emissions[:fit_chirps], sample_rates=sample_rates)
    fit_residuals_ms = {pid: round(model.residual_ms, 4) for pid, model in fitter.models.items()}

    held = emissions[held_out_index]
    emission_times_s: dict[str, float] = {}
    phones_measured: list[str] = []
    phones_missing: list[str] = []

    for pid in phones_expected:
        obs = held.obs_by_phone.get(pid)
        t_glob = fitter.to_global(pid, obs.sample_index) if obs is not None else None
        if obs is None or t_glob is None:
            phones_missing.append(pid)
            continue
        t_emit = t_glob - flight_time_s(phones[pid], room)
        emission_times_s[pid] = t_emit
        phones_measured.append(pid)

    if len(phones_measured) < 2:
        return HeldOutEvaluation(
            fit_chirp_indices=fit_indices,
            held_out_chirp_index=held_out_index,
            ref_phone_id=fitter.ref_phone_id,
            fit_residuals_ms=fit_residuals_ms,
            phones_expected=phones_expected,
            phones_measured=phones_measured,
            phones_missing=phones_missing,
            coverage_complete=False,
            emission_times_s=emission_times_s,
            per_phone_diff_ms={},
            worst_pairwise_ms=None,
            gate_passed=False,
            status="incomplete_coverage",
        )

    ref_t = (
        emission_times_s[fitter.ref_phone_id]
        if fitter.ref_phone_id in emission_times_s
        else float(np.median(list(emission_times_s.values())))
    )
    per_phone_diff_ms = {
        pid: round((t_emit - ref_t) * 1000.0, 4) for pid, t_emit in emission_times_s.items()
    }
    worst_pairwise_ms = round((max(emission_times_s.values()) - min(emission_times_s.values())) * 1000.0, 4)
    coverage_complete = len(phones_missing) == 0
    gate_passed = len(phones_measured) >= 2 and worst_pairwise_ms < 1.0
    status = "pass" if gate_passed else "fail"

    return HeldOutEvaluation(
        fit_chirp_indices=fit_indices,
        held_out_chirp_index=held_out_index,
        ref_phone_id=fitter.ref_phone_id,
        fit_residuals_ms=fit_residuals_ms,
        phones_expected=phones_expected,
        phones_measured=phones_measured,
        phones_missing=phones_missing,
        coverage_complete=coverage_complete,
        emission_times_s=emission_times_s,
        per_phone_diff_ms=per_phone_diff_ms,
        worst_pairwise_ms=worst_pairwise_ms,
        gate_passed=gate_passed,
        status=status,
    )


def fit_clocks(
    arrivals_by_phone: dict[str, list[ChirpArrival]],
    sample_rates: dict[str, int],
    phones: dict[str, PhoneConfig],
    room: RoomConfig,
    slice_end_indices: dict[str, int] | None = None,
    max_age_s: float = CLOCK_FIT_MAX_AGE_S,
) -> tuple[dict[str, ClockModel], list[EmissionRecord], ReferenceClockFitter]:
    """Pair detected arrivals across phones and fit per-phone ClockModels to the highest-SNR reference."""
    emissions = pair_arrivals_to_emissions(
        arrivals_by_phone=arrivals_by_phone,
        sample_rates=sample_rates,
        slice_end_indices=slice_end_indices,
        phones=phones,
        room=room,
    )
    fitter = ReferenceClockFitter(max_age_s=max_age_s)
    fitter.ingest_emissions(emissions, sample_rates=sample_rates)
    return fitter.models, emissions, fitter
