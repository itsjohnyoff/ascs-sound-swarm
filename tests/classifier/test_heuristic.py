"""Agent 3 phase 3 MUST: heuristic classifier > 90% clap vs drone on the simulator, < 20 ms per event."""

import time

import numpy as np
import pytest

from backend.classifier.model import SoundClassifier
from sim.benchmark import load_slices
from sim.room import judge_scenario
from sim.sources import SRC_FS, clap, drone, speech_like
from backend.dsp.replay import _parse_room
from backend.server.events import find_events_offline


def _swarm(x, rng, n=4, snr_db=20.0):
    out = {}
    for i in range(n):
        g = rng.uniform(0.2, 1.0)
        y = g * x
        noise = rng.standard_normal(len(y)) * np.sqrt(np.mean(y ** 2)) * 10 ** (-snr_db / 20)
        out[f"p{i}"] = (y + noise).astype(np.float32)
    return out


def _window(src, kind):
    n = int((0.08 if kind == "transient" else 0.75) * SRC_FS)
    pre = int(0.015 * SRC_FS) if kind == "transient" else 0
    x = np.zeros(n)
    m = min(len(src), n - pre)
    x[pre:pre + m] = src[:m]
    return x


@pytest.mark.parametrize("seed", range(10))
def test_clap_drone_speech_on_raw_sources(seed):
    rng = np.random.default_rng(seed)
    clf = SoundClassifier()
    assert clf.backend_name == "heuristic"
    assert clf.classify_swarm(_swarm(_window(clap(rng), "transient"), rng)).sound_class == "clap"
    res = clf.classify_swarm(_swarm(_window(drone(rng), "sustained"), rng))
    assert res.sound_class == "drone" and res.is_drone
    assert clf.classify_swarm(_swarm(_window(speech_like(rng), "sustained"), rng)).sound_class == "other"


def test_accuracy_and_speed_on_judge_scenarios():
    clf = SoundClassifier()
    correct = total = 0
    times = []
    for seed in range(3):
        rec = judge_scenario(seed=seed)
        meta, phones, slices = load_slices(rec)
        for det in find_events_offline(slices, phones, _parse_room(meta)):
            expected = "drone" if det.kind == "sustained" else "clap"
            t = time.perf_counter()
            res = clf.classify_swarm(det.aligned)
            times.append(time.perf_counter() - t)
            correct += res.sound_class == expected
            total += 1
            assert abs(sum(res.probs.values()) - 1) < 1e-6 and 0 <= res.confidence <= 1
    print(f"accuracy {correct}/{total}, median {1e3 * np.median(times):.1f} ms")
    assert total == 30 and correct / total > 0.9
    assert np.median(times) < 0.020
