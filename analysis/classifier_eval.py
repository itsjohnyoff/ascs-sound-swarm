"""Offline heuristic-vs-trained classifier evaluator for simulator and venue .npz recordings.

Usage:
    python -m analysis.classifier_eval data/recordings/<files>.npz [--model PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from backend.classifier.model import CLASSES, DEFAULT_MODEL_PATH, SoundClassifier
from backend.dsp.pipeline import SyncEngine
from backend.dsp.replay import load_recording_npz
from backend.server.events import TARGET_FS, find_events_offline
from sim.room import true_time

MATCH_WINDOW_S = 0.5


@dataclass
class LabeledEvent:
    source: str
    label: str
    aligned: dict[str, np.ndarray]
    fs: int


@dataclass
class BackendMetrics:
    name: str
    backend_name: str
    confusion: list[list[int]]
    per_class_f1: dict[str, float]
    macro_f1: float
    median_ms: float
    p90_ms: float
    n_events: int


def normalize_label(raw: str) -> str:
    s = raw.strip().lower()
    if s == "drone":
        return "drone"
    if s in ("clap", "claps"):
        return "clap"
    return "other"


def _infer_venue_label(meta: dict, path: Path) -> str:
    raw = meta.get("label")
    if isinstance(raw, str) and raw.strip():
        return normalize_label(raw)
    stem = path.stem.lower()
    if "drone" in stem:
        return "drone"
    if "clap" in stem:
        return "clap"
    return "other"


def _resample_to_target(samples: np.ndarray, fs: int, valid_mask: np.ndarray | None) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if valid_mask is not None:
        x = np.where(valid_mask, x, 0.0)
    if fs == TARGET_FS or len(x) == 0:
        return x
    g = math.gcd(TARGET_FS, fs)
    return resample_poly(x, TARGET_FS // g, fs // g).astype(np.float64)


def _extract_simulator_events(session) -> list[LabeledEvent]:
    truth = session.meta["truth"]
    targets = [e for e in truth["events"] if e.get("kind") != "chirp"]
    if not targets:
        return []

    engine = SyncEngine()
    engine.add_chirp_observation(session.slices, session.phones, session.room, "lfm_single")
    detections = find_events_offline(
        session.slices, session.phones, session.room, engine=engine, min_phones=2
    )
    ref = engine.ref_phone_id
    if not detections or ref is None or ref not in truth.get("phones", {}):
        return []

    out: list[LabeledEvent] = []
    for det in detections:
        local_idx = engine.to_local(ref, det.t0)
        if local_idx is None:
            continue
        t_true = true_time(truth["phones"][ref], local_idx)
        nearest = min(targets, key=lambda e: abs(e["t_s"] - t_true))
        if abs(nearest["t_s"] - t_true) > MATCH_WINDOW_S:
            continue
        out.append(LabeledEvent(session.path.name, normalize_label(nearest["kind"]), det.aligned, TARGET_FS))
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def _extract_venue_events(session) -> list[LabeledEvent]:
    label = _infer_venue_label(session.meta, session.path)
    resampled = {
        pid: _resample_to_target(s.samples_f32, s.sample_rate, getattr(s, "valid_mask", None))
        for pid, s in session.slices.items()
    }
    min_len = min((len(x) for x in resampled.values()), default=0)
    if min_len == 0:
        return []
    resampled = {pid: x[:min_len] for pid, x in resampled.items()}

    out: list[LabeledEvent] = []
    if label == "clap":
        loudest = max(resampled.values(), key=_rms)
        hop = int(0.01 * TARGET_FS)
        n_frames = min_len // hop
        if n_frames == 0:
            return []
        frame_rms = np.sqrt(np.mean(loudest[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1))
        thresh = max(6.0 * float(np.median(frame_rms)), 0.02)
        sep_frames = int(round(0.35 / 0.01))
        pre = int(0.015 * TARGET_FS)
        post = int(0.065 * TARGET_FS)

        i = 0
        while i < n_frames:
            if frame_rms[i] > thresh:
                peak_end = min(n_frames, i + 5)
                peak = i + int(np.argmax(frame_rms[i:peak_end]))
                center = peak * hop + hop // 2
                lo, hi = center - pre, center + post
                if lo >= 0 and hi <= min_len:
                    aligned = {pid: x[lo:hi] for pid, x in resampled.items()}
                    out.append(LabeledEvent(session.path.name, "clap", aligned, TARGET_FS))
                i = peak + sep_frames
            else:
                i += 1
        return out

    # Sustained / background clips ("drone" or "other"): non-overlapping 0.75 s windows.
    win = int(0.75 * TARGET_FS)
    trim = int(0.25 * TARGET_FS)
    if min_len < win:
        return []
    if min_len < 2 * trim + win:
        trim = 0

    windows: list[tuple[dict[str, np.ndarray], float]] = []
    for start in range(trim, min_len - trim - win + 1, win):
        aligned = {pid: x[start : start + win] for pid, x in resampled.items()}
        max_rms = max(_rms(x) for x in aligned.values())
        windows.append((aligned, max_rms))

    if not windows:
        return []

    if label == "drone":
        med_rms = float(np.median([r for _, r in windows]))
        cutoff = 0.2 * med_rms
        windows = [(a, r) for a, r in windows if r >= cutoff and r > 0.0]

    for aligned, _ in windows:
        out.append(LabeledEvent(session.path.name, label, aligned, TARGET_FS))
    return out


def extract_labeled_events(paths: Sequence[str | Path]) -> list[LabeledEvent]:
    events: list[LabeledEvent] = []
    for p in paths:
        session = load_recording_npz(p)
        truth = session.meta.get("truth")
        if isinstance(truth, dict) and truth.get("events"):
            events.extend(_extract_simulator_events(session))
        else:
            events.extend(_extract_venue_events(session))
    return events


def evaluate_backend(
    events: Sequence[LabeledEvent], classifier: SoundClassifier, name: str
) -> BackendMetrics:
    idx_of = {c: i for i, c in enumerate(CLASSES)}
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    times_ms: list[float] = []

    for ev in events:
        t0 = time.perf_counter()
        res = classifier.classify_swarm(ev.aligned, fs=ev.fs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        confusion[idx_of[ev.label]][idx_of[res.sound_class]] += 1

    per_class_f1: dict[str, float] = {}
    present_f1: list[float] = []
    for i, cls in enumerate(CLASSES):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(CLASSES)) if r != i)
        fn = sum(confusion[i][c] for c in range(len(CLASSES)) if c != i)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        per_class_f1[cls] = round(f1, 4)
        if sum(confusion[i]) > 0:
            present_f1.append(f1)

    macro_f1 = round(float(np.mean(present_f1)), 4) if present_f1 else 0.0
    median_ms = round(float(np.median(times_ms)), 3) if times_ms else 0.0
    p90_ms = round(float(np.percentile(times_ms, 90)), 3) if times_ms else 0.0

    return BackendMetrics(
        name=name,
        backend_name=classifier.backend_name,
        confusion=confusion,
        per_class_f1=per_class_f1,
        macro_f1=macro_f1,
        median_ms=median_ms,
        p90_ms=p90_ms,
        n_events=len(events),
    )


def evaluate_files(
    paths: Sequence[str | Path], model_path: str | Path | None = DEFAULT_MODEL_PATH
) -> tuple[BackendMetrics, BackendMetrics]:
    events = extract_labeled_events(paths)
    heuristic_clf = SoundClassifier(model_path=None)
    trained_clf = SoundClassifier(model_path=model_path)
    return (
        evaluate_backend(events, heuristic_clf, "heuristic"),
        evaluate_backend(events, trained_clf, "trained"),
    )


def _format_confusion(m: BackendMetrics) -> list[str]:
    header = "| truth \\ pred | " + " | ".join(CLASSES) + " |"
    sep = "| --- | " + " | ".join("---" for _ in CLASSES) + " |"
    rows = [
        f"| {cls} | " + " | ".join(str(v) for v in m.confusion[i]) + " |"
        for i, cls in enumerate(CLASSES)
    ]
    return [f"### Confusion Matrix: {m.name} ({m.backend_name})", "", header, sep, *rows]


def format_eval_report(heuristic: BackendMetrics, trained: BackendMetrics) -> str:
    lines = [
        "| backend | impl | events | drone F1 | clap F1 | other F1 | macro F1 | median ms/event | p90 ms/event |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in (heuristic, trained):
        lines.append(
            f"| {m.name} | {m.backend_name} | {m.n_events} "
            f"| {m.per_class_f1['drone']:.4f} | {m.per_class_f1['clap']:.4f} | {m.per_class_f1['other']:.4f} "
            f"| {m.macro_f1:.4f} | {m.median_ms:.3f} | {m.p90_ms:.3f} |"
        )
    lines.append("")
    lines.extend(_format_confusion(heuristic))
    lines.append("")
    lines.extend(_format_confusion(trained))
    lines.append("")

    if trained.backend_name == "heuristic":
        rec = "Recommendation: trained model unavailable (fell back to heuristic); using heuristic."
    elif trained.macro_f1 >= heuristic.macro_f1:
        rec = f"Recommendation: use trained ({trained.backend_name}, macro F1 {trained.macro_f1:.4f} vs {heuristic.macro_f1:.4f})."
    else:
        rec = f"Recommendation: use heuristic (macro F1 {heuristic.macro_f1:.4f} vs {trained.macro_f1:.4f})."
    lines.append(rec)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate heuristic vs trained classifier on .npz recordings.")
    ap.add_argument("recordings", nargs="+", help="One or more v2 .npz recording paths")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to trained joblib model")
    ap.add_argument("--json", action="store_true", help="Print JSON output")
    args = ap.parse_args(argv)

    heuristic, trained = evaluate_files(args.recordings, model_path=args.model)
    if args.json:
        print(json.dumps({"heuristic": asdict(heuristic), "trained": asdict(trained)}, indent=2))
    else:
        print(format_eval_report(heuristic, trained))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
