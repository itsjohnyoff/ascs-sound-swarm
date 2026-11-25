"""Tests for offline classifier evaluation on simulator and venue recordings."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.classifier_eval import (
    BackendMetrics,
    _extract_venue_events,
    evaluate_files,
    format_eval_report,
    main,
)
from backend.classifier.model import CLASSES
from shared.chirp import generate_lfm_chirp
from shared.protocol import PhoneAudioSlice
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


def test_missing_venue_class_blocks_live_recommendation() -> None:
    scores = dict(
        confusion=[[4, 0, 0], [0, 0, 0], [0, 0, 4]],
        per_class_f1={"drone": 1.0, "clap": 0.0, "other": 1.0},
        macro_f1=1.0, median_ms=1.0, p90_ms=1.0, n_events=8,
    )
    heuristic = BackendMetrics("heuristic", "heuristic", **scores)
    trained = BackendMetrics("trained", "hgb", **scores)

    assert "missing usable examples for clap" in format_eval_report(heuristic, trained)
    assert "Recommendation: use trained" not in format_eval_report(heuristic, trained)


def test_uncalibrated_clap_uses_a_single_phone(tmp_path: Path) -> None:
    fs = 48000
    early = np.zeros(fs, dtype=np.float32)
    late = np.zeros(fs, dtype=np.float32)
    early[fs // 4 : fs // 4 + 480] = 0.9
    late[fs // 4 + 4800 : fs // 4 + 5280] = 0.8
    session = SimpleNamespace(
        path=tmp_path / "venue-claps.npz", meta={},
        slices={"early": PhoneAudioSlice("early", early, 0, fs),
                "late": PhoneAudioSlice("late", late, 0, fs)},
    )

    [event] = _extract_venue_events(session)
    assert set(event.aligned) == {"early"}
    assert np.max(event.aligned["early"]) > 0.8


def test_venue_clip_with_calibration_chirp_is_rejected(tmp_path: Path) -> None:
    fs = 48000
    samples = np.zeros(2 * fs, dtype=np.float32)
    chirp = generate_lfm_chirp(fs=fs)
    samples[fs : fs + len(chirp)] = chirp
    session = SimpleNamespace(
        path=tmp_path / "venue-noise.npz", meta={},
        slices={"p1": PhoneAudioSlice("p1", samples, 0, fs)},
    )

    with pytest.raises(ValueError, match="calibration chirp"):
        _extract_venue_events(session)
