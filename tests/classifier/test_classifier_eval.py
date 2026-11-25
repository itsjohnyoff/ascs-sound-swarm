"""Tests for offline classifier evaluation on simulator and venue recordings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.classifier_eval import (
    evaluate_files,
    format_eval_report,
    main,
)
from backend.classifier.model import CLASSES
from sim.room import DEFAULT_ROOM, SimEvent, default_phones, judge_scenario, save_npz, simulate


def _strip_truth(rec: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    meta = json.loads(str(rec["meta"]))
    meta.pop("truth", None)
    meta.pop("simulated", None)
    rec["meta"] = np.asarray(json.dumps(meta))
    return rec


def _make_venue_clips(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(42)
    phones = default_phones(rng, n=4, snr_db=28.0)

    clap_events = [SimEvent(t, 3.5, 3.0, "clap") for t in (1.0, 2.5, 4.0, 5.5)]
    claps_rec = _strip_truth(simulate(DEFAULT_ROOM, phones, clap_events, 7.0, seed=1))
    claps_path = save_npz(claps_rec, tmp_path / "2026-11-26-venue-claps.npz")

    noise_rec = _strip_truth(simulate(DEFAULT_ROOM, phones, [], 4.0, seed=2))
    noise_path = save_npz(noise_rec, tmp_path / "2026-11-26-venue-noise.npz")

    drone_events = [SimEvent(0.5, 4.0, 3.0, "drone", duration_s=3.5)]
    drone_rec = _strip_truth(simulate(DEFAULT_ROOM, phones, drone_events, 4.5, seed=3))
    drone_path = save_npz(drone_rec, tmp_path / "2026-11-26-venue-drone.npz")

    return [claps_path, noise_path, drone_path]


def test_eval_on_simulator_judge_recording(tmp_path: Path) -> None:
    path = save_npz(judge_scenario(seed=0), tmp_path / "sim-judge.npz")
    heuristic, trained = evaluate_files([path])

    assert heuristic.n_events >= 8
    assert trained.n_events == heuristic.n_events
    for m in (heuristic, trained):
        assert 0.0 <= m.macro_f1 <= 1.0
        assert m.median_ms > 0.0
        assert m.p90_ms >= m.median_ms
        assert len(m.confusion) == len(CLASSES)
        assert all(len(row) == len(CLASSES) for row in m.confusion)
        assert sum(sum(row) for row in m.confusion) == m.n_events

    report = format_eval_report(heuristic, trained)
    for token in ("macro F1", "ms/event", "drone", "clap", "other"):
        assert token in report


def test_eval_on_uncalibrated_venue_style_npz_files(tmp_path: Path) -> None:
    paths = _make_venue_clips(tmp_path)
    heuristic, trained = evaluate_files(paths)

    assert heuristic.n_events > 0
    assert trained.n_events == heuristic.n_events
    for m in (heuristic, trained):
        assert 0.0 <= m.macro_f1 <= 1.0
        assert m.median_ms > 0.0
        row_sums = [sum(row) for row in m.confusion]
        assert all(count > 0 for count in row_sums), f"expected all 3 classes in {m.confusion}"
        assert sum(row_sums) == m.n_events


def test_eval_missing_model_falls_back_gracefully(tmp_path: Path) -> None:
    paths = _make_venue_clips(tmp_path)
    heuristic, trained = evaluate_files(paths, model_path=tmp_path / "missing.joblib")

    assert heuristic.backend_name == "heuristic"
    assert trained.backend_name == "heuristic"
    assert trained.n_events == heuristic.n_events


def test_eval_cli_json_and_markdown(tmp_path: Path, capsys) -> None:
    paths = _make_venue_clips(tmp_path)
    str_paths = [str(p) for p in paths]

    rc_md = main(str_paths)
    out_md = capsys.readouterr().out
    assert rc_md == 0
    assert "macro F1" in out_md

    rc_json = main([*str_paths, "--json"])
    out_json = capsys.readouterr().out
    assert rc_json == 0
    payload = json.loads(out_json)
    assert "heuristic" in payload
    assert "trained" in payload
    assert payload["heuristic"]["n_events"] == payload["trained"]["n_events"]
