"""Shared v2 .npz recording loader, validator, and held-out chirp replay CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from shared.protocol import CHIRP_DURATION_S, PhoneAudioSlice, PhoneConfig, RoomConfig
from backend.dsp.clock import (
    AudioSliceV2,
    ChirpArrival,
    EmissionRecord,
    HeldOutEvaluation,
    ReferenceClockFitter,
    evaluate_held_out_chirp,
    flight_time_s,
    make_phone_slice,
    pair_arrivals_to_emissions,
)
from backend.dsp.pipeline import LocalizationPipeline
from backend.dsp.sync import SyncEngine, detect_all_chirps
from backend.dsp.trigger import OnsetDetector


class RecordingValidationError(ValueError):
    """Raised when a v2 .npz recording file is missing required keys or contains invalid data."""


@dataclass(slots=True)
class RecordedSession:
    path: Path
    meta: dict[str, Any]
    room: RoomConfig
    phones: dict[str, PhoneConfig]
    slices: dict[str, PhoneAudioSlice | AudioSliceV2]


@dataclass(slots=True)
class ReplayEvent:
    """Single post-chirp sound event localized during offline replay."""

    event_id: str
    timestamp_s: float
    x_m: float
    y_m: float
    uncertainty_m: float
    residual_ms: float
    confidence: float
    runtime_ms: float
    phones_used: list[str]
    inlier_pairs: int
    total_pairs: int


@dataclass(slots=True)
class ReplayReport:
    path: str
    phones_loaded: list[str]
    sample_rates: dict[str, int]
    arrivals_per_phone: dict[str, int]
    emissions_paired: int
    held_out: HeldOutEvaluation
    full_fit_ref_phone_id: str | None
    full_fit_residuals_ms: dict[str, float]
    full_fit_drift_ppm: dict[str, float]
    events: list[ReplayEvent] = field(default_factory=list)


def _extract_scalar_int(arr: np.ndarray, key: str) -> int:
    if arr.ndim != 0 and arr.size != 1:
        raise RecordingValidationError(f"Key '{key}' must be a scalar integer, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise RecordingValidationError(f"Key '{key}' must have integer dtype, got {arr.dtype}")
    return int(arr.reshape(-1)[0])


def _extract_meta_str(arr: np.ndarray) -> str:
    if arr.dtype.kind == "O":
        raise RecordingValidationError("Key 'meta' uses object dtype; pickle-backed arrays are disallowed")
    if arr.ndim == 0:
        val = arr.item()
    elif arr.size == 1:
        val = arr.reshape(-1)[0]
    else:
        raise RecordingValidationError(f"Key 'meta' must be a scalar JSON string, got shape {arr.shape}")
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, str):
        return val
    raise RecordingValidationError(f"Key 'meta' must be a JSON string, got {type(val).__name__}")


def _parse_room(meta: dict[str, Any]) -> RoomConfig:
    room_raw = meta.get("room")
    if not isinstance(room_raw, dict):
        raise RecordingValidationError("meta JSON must contain a 'room' object")

    # Accept either speaker_x_m/speaker_y_m directly in room or meta['speaker'] = {'x_m': ..., 'y_m': ...}
    speaker = meta.get("speaker")
    width_m = float(room_raw.get("width_m", 8.0))
    height_m = float(room_raw.get("height_m", 6.0))
    temperature_c = float(meta.get("temperature_c", room_raw.get("temperature_c", 20.0)))
    if isinstance(speaker, dict):
        spk_x = float(speaker.get("x_m", room_raw.get("speaker_x_m", width_m * 0.5)))
        spk_y = float(speaker.get("y_m", room_raw.get("speaker_y_m", 0.5)))
    else:
        spk_x = float(room_raw.get("speaker_x_m", width_m * 0.5))
        spk_y = float(room_raw.get("speaker_y_m", 0.5))

    try:
        return RoomConfig(
            width_m=width_m,
            height_m=height_m,
            temperature_c=temperature_c,
            speaker_x_m=spk_x,
            speaker_y_m=spk_y,
        )
    except Exception as exc:
        raise RecordingValidationError(f"Invalid room metadata in 'meta': {exc}") from exc


def _parse_phone_positions(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_phones = meta.get("phones")
    if isinstance(raw_phones, dict):
        out: dict[str, dict[str, Any]] = {}
        for pid, info in raw_phones.items():
            if not isinstance(info, dict):
                raise RecordingValidationError(f"meta.phones['{pid}'] must be an object")
            out[str(pid)] = info
        return out
    if isinstance(raw_phones, list):
        out = {}
        for idx, item in enumerate(raw_phones):
            if not isinstance(item, dict) or "phone_id" not in item:
                raise RecordingValidationError(f"meta.phones[{idx}] must be an object with 'phone_id'")
            out[str(item["phone_id"])] = item
        return out
    raise RecordingValidationError("meta JSON must contain a 'phones' mapping or list")


def load_recording_npz(path: str | Path) -> RecordedSession:
    """Load and validate a v2 .npz recording without pickle."""
    p = Path(path)
    if not p.exists():
        raise RecordingValidationError(f"Recording file not found: {p}")
    if not p.is_file():
        raise RecordingValidationError(f"Recording path is not a file: {p}")

    try:
        npz = np.load(p, allow_pickle=False)
    except ValueError as exc:
        raise RecordingValidationError(
            f"Refusing to load '{p}' (object/pickle arrays are disallowed: {exc})"
        ) from exc
    except Exception as exc:
        raise RecordingValidationError(f"Failed to read .npz archive '{p}': {exc}") from exc

    with npz:
        def _read_key(k: str) -> np.ndarray:
            try:
                return npz[k]
            except ValueError as exc:
                raise RecordingValidationError(
                    f"Refusing to load '{p}' key '{k}' (object/pickle arrays are disallowed: {exc})"
                ) from exc

        keys = set(npz.files)
        if "meta" not in keys:
            raise RecordingValidationError("Recording is missing required 'meta' JSON key")

        meta_str = _extract_meta_str(_read_key("meta"))
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError as exc:
            raise RecordingValidationError(f"Key 'meta' is not valid JSON: {exc}") from exc
        if not isinstance(meta, dict):
            raise RecordingValidationError("Key 'meta' must decode to a JSON object")

        room = _parse_room(meta)
        phone_meta = _parse_phone_positions(meta)

        pcm_ids = sorted(k[: -len("_pcm")] for k in keys if k.endswith("_pcm"))
        if not pcm_ids:
            raise RecordingValidationError("Recording contains no '<id>_pcm' arrays")

        phones: dict[str, PhoneConfig] = {}
        slices: dict[str, PhoneAudioSlice | AudioSliceV2] = {}

        for pid in pcm_ids:
            req_keys = (f"{pid}_pcm", f"{pid}_start_index", f"{pid}_fs", f"{pid}_valid")
            missing = [rk for rk in req_keys if rk not in keys]
            if missing:
                raise RecordingValidationError(f"Phone '{pid}' is missing required keys: {', '.join(missing)}")
            if pid not in phone_meta:
                raise RecordingValidationError(f"Phone '{pid}' has audio arrays but no entry in meta.phones")

            pcm = _read_key(f"{pid}_pcm")
            if pcm.ndim != 1 or len(pcm) == 0:
                raise RecordingValidationError(f"'{pid}_pcm' must be a non-empty 1D array, got shape {pcm.shape}")
            if not np.issubdtype(pcm.dtype, np.floating):
                raise RecordingValidationError(f"'{pid}_pcm' must be floating-point (float32), got {pcm.dtype}")
            pcm_f32 = np.ascontiguousarray(pcm, dtype=np.float32)

            start_idx = _extract_scalar_int(_read_key(f"{pid}_start_index"), f"{pid}_start_index")
            fs = _extract_scalar_int(_read_key(f"{pid}_fs"), f"{pid}_fs")
            if not (8000 <= fs <= 192000):
                raise RecordingValidationError(f"'{pid}_fs' out of valid range [8000, 192000]: {fs}")

            valid = _read_key(f"{pid}_valid")
            if valid.ndim != 1 or len(valid) != len(pcm_f32):
                raise RecordingValidationError(
                    f"'{pid}_valid' length ({valid.shape}) must match '{pid}_pcm' ({pcm_f32.shape})"
                )
            if valid.dtype != np.bool_:
                raise RecordingValidationError(f"'{pid}_valid' must have bool dtype, got {valid.dtype}")
            valid_bool = np.ascontiguousarray(valid, dtype=bool)

            if np.any(valid_bool) and not np.all(np.isfinite(pcm_f32[valid_bool])):
                raise RecordingValidationError(f"'{pid}_pcm' contains NaN/Inf in valid samples")

            pm = phone_meta[pid]
            if "x_m" not in pm or "y_m" not in pm:
                raise RecordingValidationError(f"meta.phones['{pid}'] must specify 'x_m' and 'y_m'")

            phones[pid] = PhoneConfig(
                phone_id=pid,
                label=str(pm.get("label", pid)),
                x_m=float(pm["x_m"]),
                y_m=float(pm["y_m"]),
                sample_rate=fs,
            )
            slices[pid] = make_phone_slice(
                phone_id=pid,
                samples_f32=pcm_f32,
                start_sample_index=start_idx,
                sample_rate=fs,
                valid_mask=valid_bool,
            )

    return RecordedSession(path=p, meta=meta, room=room, phones=phones, slices=slices)


def replay_localize_events(
    session: RecordedSession,
    fitter: ReferenceClockFitter,
    emissions: list[EmissionRecord],
    target_fs: int = 48000,
    cooldown_s: float = 0.40,
) -> list[ReplayEvent]:
    """Run OnsetDetector + SyncEngine + LocalizationPipeline over post-chirp audio in a recording."""
    if len(fitter.models) < 2:
        return []

    engine = SyncEngine()
    engine.load_fitter(fitter)
    pipeline = LocalizationPipeline()

    chirp_windows: list[tuple[float, float]] = []
    for em in emissions:
        emit_ts: list[float] = []
        for pid, obs in em.obs_by_phone.items():
            if pid not in session.phones:
                continue
            t_glob = engine.to_global(pid, obs.sample_index)
            if t_glob is not None:
                emit_ts.append(t_glob - flight_time_s(session.phones[pid], session.room))
        if emit_ts:
            t_emit = float(np.median(emit_ts))
            chirp_windows.append((t_emit - 0.15, t_emit + CHIRP_DURATION_S + 0.35))

    def _in_chirp(t_s: float) -> bool:
        return any(lo <= t_s <= hi for lo, hi in chirp_windows)

    raw_onsets: list[tuple[float, str]] = []
    sync_ends: list[float] = []
    for pid, slc in session.slices.items():
        if not engine.is_synced(pid) or pid not in session.phones:
            continue
        if len(slc.samples_f32) >= 4:
            t_end = engine.to_global(pid, slc.start_sample_index + len(slc.samples_f32) - 2)
            if t_end is not None:
                sync_ends.append(t_end)
        det = OnsetDetector(refractory_s=cooldown_s)
        idx_hits = det.push(
            slc.samples_f32,
            start_sample_index=slc.start_sample_index,
            fs=slc.sample_rate,
            valid_mask=getattr(slc, "valid_mask", None),
        )
        for idx in idx_hits:
            t_on = engine.to_global(pid, idx)
            if t_on is None or _in_chirp(t_on):
                continue
            raw_onsets.append((t_on, pid))

    if not raw_onsets:
        return []

    raw_onsets.sort(key=lambda item: item[0])
    cluster_window_s = 0.06
    clusters: list[list[tuple[float, str]]] = []
    cur_cluster: list[tuple[float, str]] = [raw_onsets[0]]
    group_start = raw_onsets[0][0]
    for item in raw_onsets[1:]:
        if item[0] - group_start <= cluster_window_s:
            cur_cluster.append(item)
        else:
            clusters.append(cur_cluster)
            cur_cluster = [item]
            group_start = item[0]
    clusters.append(cur_cluster)

    max_global_end = float(np.median(sync_ends)) if sync_ends else None
    events: list[ReplayEvent] = []
    last_event_t = -float("inf")

    for cluster in clusters:
        distinct_phones = {pid for _, pid in cluster}
        if len(distinct_phones) < 2:
            continue
        t_event = min(t for t, _ in cluster)
        if t_event - last_event_t < cooldown_s:
            continue

        win_start = t_event - 0.02
        next_chirp_lo = min((lo for lo, _ in chirp_windows if lo > t_event), default=None)
        win_hi = next_chirp_lo - 0.01 if next_chirp_lo is not None else float("inf")
        if max_global_end is not None:
            win_hi = min(win_hi, max_global_end)
        if win_hi < t_event + 0.08:
            continue

        aligned_short = engine.extract_aligned_window(
            session.slices,
            win_start,
            min(win_hi, t_event + 0.12),
            target_fs=target_fs,
        )
        long_end = min(win_hi, t_event + 0.65)
        if long_end > min(win_hi, t_event + 0.12) + 0.05:
            aligned_long = engine.extract_aligned_window(
                session.slices,
                win_start,
                long_end,
                target_fs=target_fs,
            )
            aligned = (
                aligned_long
                if len(aligned_long) >= len(aligned_short) and len(aligned_long) >= 2
                else aligned_short
            )
        else:
            aligned = aligned_short

        if len(aligned) < 2:
            continue

        t0 = time.perf_counter()
        res = pipeline.detect_and_localize(
            aligned,
            session.phones,
            session.room,
            fs=target_fs,
            timestamp_s=t_event,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if res is None:
            continue

        last_event_t = t_event
        events.append(
            ReplayEvent(
                event_id=res.event_id,
                timestamp_s=round(t_event, 4),
                x_m=res.x_m,
                y_m=res.y_m,
                uncertainty_m=res.uncertainty_m,
                residual_ms=res.residual_ms,
                confidence=res.confidence,
                runtime_ms=round(dt_ms, 2),
                phones_used=list(res.phones_used),
                inlier_pairs=sum(1 for p in res.pair_tdoas_ms if p.inlier),
                total_pairs=len(res.pair_tdoas_ms),
            )
        )

    return events


def replay_recording(
    path: str | Path,
    fit_chirps: int = 2,
    held_out_index: int = 2,
    localize: bool = False,
    target_fs: int = 48000,
    cooldown_s: float = 0.40,
) -> ReplayReport:
    """Replay a saved v2 .npz recording: detect chirps, evaluate held-out sync, and run full fit."""
    session = load_recording_npz(path)

    arrivals_by_phone: dict[str, list[ChirpArrival]] = {}
    sample_rates: dict[str, int] = {}
    slice_ends: dict[str, int] = {}

    for pid, slc in session.slices.items():
        arrs = detect_all_chirps(
            slc.samples_f32,
            fs=slc.sample_rate,
            start_sample_index=slc.start_sample_index,
            valid_mask=slc.valid_mask,
        )
        arrivals_by_phone[pid] = arrs
        sample_rates[pid] = slc.sample_rate
        slice_ends[pid] = slc.start_sample_index + len(slc.samples_f32)

    emissions = pair_arrivals_to_emissions(
        arrivals_by_phone=arrivals_by_phone,
        sample_rates=sample_rates,
        slice_end_indices=slice_ends,
        phones=session.phones,
        room=session.room,
    )

    held_out = evaluate_held_out_chirp(
        emissions=emissions,
        phones=session.phones,
        room=session.room,
        sample_rates=sample_rates,
        fit_chirps=fit_chirps,
        held_out_index=held_out_index,
    )

    full_fitter = ReferenceClockFitter()
    full_fitter.ingest_emissions(emissions, sample_rates=sample_rates)
    full_fit_residuals = {pid: round(m.residual_ms, 4) for pid, m in full_fitter.models.items()}
    full_fit_drift = {pid: round(m.drift_ppm, 3) for pid, m in full_fitter.models.items()}
    events = (
        replay_localize_events(
            session,
            full_fitter,
            emissions,
            target_fs=target_fs,
            cooldown_s=cooldown_s,
        )
        if localize
        else []
    )

    return ReplayReport(
        path=str(session.path),
        phones_loaded=sorted(session.phones.keys()),
        sample_rates=sample_rates,
        arrivals_per_phone={pid: len(arrs) for pid, arrs in arrivals_by_phone.items()},
        emissions_paired=len(emissions),
        held_out=held_out,
        full_fit_ref_phone_id=full_fitter.ref_phone_id,
        full_fit_residuals_ms=full_fit_residuals,
        full_fit_drift_ppm=full_fit_drift,
        events=events,
    )


def _format_report(report: ReplayReport) -> str:
    ho = report.held_out
    lines = [
        f"ASCS v2 Replay: {report.path}",
        f"Phones ({len(report.phones_loaded)}): "
        + ", ".join(
            f"{pid} ({report.sample_rates[pid]} Hz, {report.arrivals_per_phone[pid]} chirps)"
            for pid in report.phones_loaded
        ),
        f"Paired emissions: {report.emissions_paired} | Fit reference: {ho.ref_phone_id or 'none'}",
        "Fit residuals (chirps "
        + ",".join(str(i + 1) for i in ho.fit_chirp_indices)
        + "): "
        + (", ".join(f"{pid}={ms:.4f} ms" for pid, ms in sorted(ho.fit_residuals_ms.items())) or "none"),
        f"Held-out chirp #{ho.held_out_chirp_index + 1} (scored BEFORE fitting):",
        f"  Status: {ho.status} | Gate (<1.000 ms, >=2 phones): {'PASS' if ho.gate_passed else 'FAIL'}",
        f"  Measured phones: {ho.phones_measured} | Missing: {ho.phones_missing} | Complete coverage: {ho.coverage_complete}",
        "  Per-phone diff: "
        + (
            ", ".join(f"{pid}={ms:+.4f} ms" for pid, ms in sorted(ho.per_phone_diff_ms.items()))
            or "insufficient coverage"
        ),
        f"  Worst pairwise disagreement: {ho.worst_pairwise_ms:.4f} ms"
        if ho.worst_pairwise_ms is not None
        else "  Worst pairwise disagreement: N/A",
        "Full multi-chirp fit drift (ppm): "
        + (", ".join(f"{pid}={ppm:+.2f} ppm" for pid, ppm in sorted(report.full_fit_drift_ppm.items())) or "none"),
    ]
    if report.events:
        runtimes = [ev.runtime_ms for ev in report.events]
        lines.append(
            f"Localized post-chirp events ({len(report.events)}): "
            f"mean_runtime={np.mean(runtimes):.2f} ms, max_runtime={np.max(runtimes):.2f} ms"
        )
        for ev in report.events:
            lines.append(
                f"  {ev.event_id} t={ev.timestamp_s:.3f}s -> ({ev.x_m:.2f}, {ev.y_m:.2f}) m "
                f"| res={ev.residual_ms:.3f} ms | unc={ev.uncertainty_m:.2f} m "
                f"| pairs={ev.inlier_pairs}/{ev.total_pairs} | phones={len(ev.phones_used)} "
                f"| runtime={ev.runtime_ms:.2f} ms"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a v2 .npz recording and score held-out chirp sync.")
    parser.add_argument("recording", help="Path to v2 .npz recording")
    parser.add_argument("--fit-chirps", type=int, default=2, help="Number of initial chirps used for clock fit")
    parser.add_argument("--held-out-index", type=int, default=2, help="0-based chirp index scored before fitting")
    parser.add_argument("--localize", action="store_true", help="Run 2D localization replay on post-chirp audio")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    try:
        report = replay_recording(
            args.recording,
            fit_chirps=args.fit_chirps,
            held_out_index=args.held_out_index,
            localize=args.localize,
        )
    except RecordingValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(_format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
