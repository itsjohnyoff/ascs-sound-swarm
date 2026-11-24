"""Shared v2 .npz recording loader, validator, and held-out chirp replay CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from shared.protocol import PhoneAudioSlice, PhoneConfig, RoomConfig
from backend.dsp.clock import (
    AudioSliceV2,
    ChirpArrival,
    HeldOutEvaluation,
    ReferenceClockFitter,
    evaluate_held_out_chirp,
    make_phone_slice,
    pair_arrivals_to_emissions,
)
from backend.dsp.sync import detect_all_chirps


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


def replay_recording(
    path: str | Path,
    fit_chirps: int = 2,
    held_out_index: int = 2,
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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a v2 .npz recording and score held-out chirp sync.")
    parser.add_argument("recording", help="Path to v2 .npz recording")
    parser.add_argument("--fit-chirps", type=int, default=2, help="Number of initial chirps used for clock fit")
    parser.add_argument("--held-out-index", type=int, default=2, help="0-based chirp index scored before fitting")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    try:
        report = replay_recording(
            args.recording,
            fit_chirps=args.fit_chirps,
            held_out_index=args.held_out_index,
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
