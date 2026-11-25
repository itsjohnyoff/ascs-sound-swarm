"""Train the drone / clap / other model (Agent 3, phase 4).

    python -m backend.classifier.train                         # synthetic only, no network
    python -m backend.classifier.train --droneaudioset 4       # + 4 streamed DroneAudioSet recordings
    python -m backend.classifier.train --local data/train      # + venue recordings (default path)

Writes `models/drone_classifier.joblib` (loaded by `SoundClassifier` when present) and
`models/metrics.json`. Split is by source recording (GroupShuffleSplit), so held-out
scores never see a window of a training recording. The shipped model is refit on all data.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from backend.classifier import dataset as D
from backend.classifier.features import MODEL_FEATURES

log = logging.getLogger("train")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=10,
                                          l2_regularization=0.1, class_weight="balanced", random_state=seed)


def build(args, rng: np.random.Generator) -> D.Dataset:
    data, noise_refs = D.local(args.local, rng)
    if len(data):
        log.info("local: %d windows from %s", len(data), args.local)
    data.extend(D.synthetic(args.synthetic, rng, noise_refs=noise_refs or None))
    if args.droneaudioset:
        try:
            data.extend(D.droneaudioset(args.droneaudioset, args.das_windows, rng))
        except Exception as exc:  # noqa: BLE001 - offline or missing extra: train on what we have
            log.warning("DroneAudioSet skipped: %s", exc)
    return data


def judge_check(bundle: dict, seeds=(0, 1, 2)) -> dict:
    """End-to-end on the simulator: event finder -> aligned windows -> fused classifier."""
    from backend.classifier.model import SoundClassifier
    from backend.dsp.replay import _parse_room
    from backend.server.events import find_events_offline
    from sim.benchmark import load_slices
    from sim.room import judge_scenario

    clf = SoundClassifier(model_path=None)
    clf._bundle, clf.backend_name = bundle, bundle["backend_name"]
    correct = total = 0
    times = []
    for seed in seeds:
        meta, phones, slices = load_slices(judge_scenario(seed=seed))
        for det in find_events_offline(slices, phones, _parse_room(meta)):
            t = time.perf_counter()
            res = clf.classify_swarm(det.aligned)
            times.append(time.perf_counter() - t)
            correct += res.sound_class == ("drone" if det.kind == "sustained" else "clap")
            total += 1
    return {"events": total, "accuracy": correct / max(total, 1), "median_ms": 1e3 * float(np.median(times)),
            "p90_ms": 1e3 * float(np.percentile(times, 90))}


def train(args) -> dict:
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    X, y, groups, origin = build(args, rng).arrays()
    log.info("%d windows (%s), %d groups, features in %.0f s", len(y),
             {c: int((y == c).sum()) for c in D.CLASSES}, len(set(groups)), time.time() - t0)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed).split(X, y, groups))
    model = make_model(args.seed).fit(X[tr], y[tr])
    pred = model.predict(X[te])
    labels = list(D.CLASSES)
    held = {
        "macro_f1": float(f1_score(y[te], pred, average="macro", labels=labels)),
        "report": classification_report(y[te], pred, labels=labels, output_dict=True, zero_division=0),
        "confusion": {"labels": labels, "matrix": confusion_matrix(y[te], pred, labels=labels).tolist()},
        "by_origin": {},
    }
    for o in sorted(set(origin[te].tolist())):
        m = origin[te] == o
        held["by_origin"][str(o)] = {"n": int(m.sum()), "accuracy": float(np.mean(pred[m] == y[te][m]))}

    final = make_model(args.seed).fit(X, y)
    bundle = {"model": final, "features": list(MODEL_FEATURES), "classes": list(final.classes_),
              "backend_name": "hgb", "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sim = {"skipped": True} if args.skip_sim else judge_check(bundle)

    metrics = {
        "trained_at": bundle["trained_at"],
        "model": "HistGradientBoostingClassifier",
        "features": list(MODEL_FEATURES),
        "data": {"windows": len(y), "groups": len(set(groups)),
                 "per_class": {c: int((y == c).sum()) for c in labels},
                 "per_origin": {str(o): int((origin == o).sum()) for o in sorted(set(origin.tolist()))},
                 "args": {k: v for k, v in vars(args).items() if k != "out"}},
        "split": {"method": "GroupShuffleSplit by source recording", "train": int(len(tr)), "test": int(len(te)),
                  "test_groups": len(set(groups[te]))},
        "held_out": held,
        "sim_judge_scenarios": sim,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(bundle, out / "drone_classifier.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    size_kb = (out / "drone_classifier.joblib").stat().st_size / 1024
    log.info("held-out macro F1 %.3f %s | sim judge %s | model %.0f kB", held["macro_f1"], held["by_origin"],
             sim, size_kb)
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", type=int, default=400, help="synthetic windows per class")
    p.add_argument("--droneaudioset", type=int, default=0, help="DroneAudioSet recordings to stream (0 = none)")
    p.add_argument("--das-windows", type=int, default=80, help="windows cut from each DroneAudioSet recording")
    p.add_argument("--local", default="data/train", help="venue recordings: <dir>/<drone|clap|other|noise>/*.wav")
    p.add_argument("--out", default=str(MODELS_DIR))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-sim", action="store_true", help="skip the end-to-end simulator check (tests)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
