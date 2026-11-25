"""Agent 3 phase 4: trained classifier loads when present, falls back to heuristics otherwise."""

import json
import time

import numpy as np
import pytest

from backend.classifier import train
from backend.classifier.model import DEFAULT_MODEL_PATH, SoundClassifier
from sim.augment import place_in_window
from sim.sources import SRC_FS, TRAINING_SOURCES, clap, drone

METRICS = DEFAULT_MODEL_PATH.with_name("metrics.json")


def _swarm(x, rng, n=4, snr_db=20.0):
    out = {}
    for i in range(n):
        y = rng.uniform(0.2, 1.0) * x
        out[f"p{i}"] = (y + rng.standard_normal(len(y)) * np.sqrt(np.mean(y ** 2)) * 10 ** (-snr_db / 20)).astype(np.float32)
    return out


def _cases(rng):
    """(window, expected) for every training source, in the window kind the server would use."""
    kinds = {"clap": "transient", "knock": "transient"}
    for label, fns in TRAINING_SOURCES.items():
        for fn in fns:
            src = fn(rng) if fn.__name__ in kinds else fn(rng, duration_s=1.2)
            yield place_in_window(src, kinds.get(fn.__name__, "sustained"), rng, 30.0), label, fn.__name__


def test_missing_or_broken_model_falls_back(tmp_path):
    assert SoundClassifier(model_path=tmp_path / "nope.joblib").backend_name == "heuristic"
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    clf = SoundClassifier(model_path=bad)
    assert clf.backend_name == "heuristic"
    rng = np.random.default_rng(0)
    assert clf.classify_swarm(_swarm(place_in_window(clap(rng), "transient", rng, 30.0), rng)).sound_class == "clap"


def test_tiny_training_roundtrip(tmp_path):
    assert train.main(["--synthetic", "60", "--out", str(tmp_path), "--local", str(tmp_path / "none"),
                       "--skip-sim"]) == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["split"]["method"].startswith("GroupShuffleSplit")
    assert metrics["held_out"]["macro_f1"] > 0.7
    clf = SoundClassifier(model_path=tmp_path / "drone_classifier.joblib")
    assert clf.backend_name == "hgb"
    rng = np.random.default_rng(1)
    res = clf.classify_swarm(_swarm(place_in_window(drone(rng, 1.2), "sustained", rng, 30.0), rng))
    assert abs(sum(res.probs.values()) - 1) < 1e-6 and res.backend_name == "hgb"


shipped = pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason="no trained model in models/")


@shipped
def test_shipped_model_metrics_and_size():
    assert DEFAULT_MODEL_PATH.stat().st_size < 10 * 1024 * 1024
    m = json.loads(METRICS.read_text())
    assert m["held_out"]["macro_f1"] > 0.9
    assert m["sim_judge_scenarios"]["accuracy"] > 0.9
    assert SoundClassifier().backend_name == "hgb"


@shipped
def test_shipped_model_on_every_source():
    clf = SoundClassifier()
    correct, total, times, wrong = 0, 0, [], []
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        for window, label, name in _cases(rng):
            aligned = _swarm(window, rng)
            t = time.perf_counter()
            res = clf.classify_swarm(aligned, fs=SRC_FS)
            times.append(time.perf_counter() - t)
            total += 1
            correct += res.sound_class == label
            if res.sound_class != label:
                wrong.append((name, res.sound_class))
    print(f"accuracy {correct}/{total}, median {1e3 * np.median(times):.1f} ms, wrong {wrong}")
    assert correct / total > 0.9
    assert np.median(times) < 0.020
