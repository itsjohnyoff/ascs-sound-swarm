"""Tests for backend/dsp/tune.py grid-search tuning harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.dsp.tune import (
    GroundTruthEvent,
    TuneParams,
    TuneRow,
    _match_errors,
    default_grid,
    format_tuning_markdown,
    load_ground_truth,
    main,
    sweep_recording,
)
from sim.room import judge_scenario, save_npz


def test_load_ground_truth_csv_and_meta(tmp_path: Path) -> None:
    # 1. judge_eval.py CSV format
    judge_csv = tmp_path / "judge.csv"
    with judge_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["trial_idx", "label", "expected_class", "gt_x_m", "gt_y_m", "timestamp_s", "t_s"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_idx": 1,
                "label": "center",
                "expected_class": "clap",
                "gt_x_m": "3.0",
                "gt_y_m": "2.5",
                "timestamp_s": "1760000000.8",
                "t_s": "14.0",
            }
        )
        writer.writerow(
            {
                "trial_idx": 2,
                "label": "corner",
                "expected_class": "drone",
                "gt_x_m": "5.5",
                "gt_y_m": "3.5",
                "timestamp_s": "1760000003.3",
                "t_s": "16.5",
            }
        )
    gt_judge = load_ground_truth({}, truth_csv=judge_csv)
    assert len(gt_judge) == 2
    assert gt_judge[0] == GroundTruthEvent(x_m=3.0, y_m=2.5, t_s=14.0, kind="clap", label="center")
    assert gt_judge[1] == GroundTruthEvent(x_m=5.5, y_m=3.5, t_s=16.5, kind="drone", label="corner")

    judge_csv.write_text("trial_idx,gt_x_m,gt_y_m,timestamp_s\n1,3,2,1760000000.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recording-relative t_s"):
        load_ground_truth({}, truth_csv=judge_csv)

    # 2. Minimal x_m,y_m CSV format
    min_csv = tmp_path / "minimal.csv"
    min_csv.write_text("x_m,y_m\n1.5,2.0\n4.2,3.1\n", encoding="utf-8")
    gt_min = load_ground_truth({}, truth_csv=min_csv)
    assert len(gt_min) == 2
    assert (gt_min[0].x_m, gt_min[0].y_m, gt_min[0].t_s) == (1.5, 2.0, None)
    assert (gt_min[1].x_m, gt_min[1].y_m, gt_min[1].t_s) == (4.2, 3.1, None)

    # 3. meta["clap_points"]
    meta_claps = {
        "clap_points": [
            {"x_m": 2.0, "y_m": 3.0, "label": "pA"},
            [6.0, 4.0],
        ]
    }
    gt_claps = load_ground_truth(meta_claps)
    assert len(gt_claps) == 2
    assert (gt_claps[0].x_m, gt_claps[0].y_m, gt_claps[0].label) == (2.0, 3.0, "pA")
    assert (gt_claps[1].x_m, gt_claps[1].y_m) == (6.0, 4.0)

    # 4. meta["points"]
    meta_pts = {"points": [{"x": 1.0, "y": 1.5}]}
    gt_pts = load_ground_truth(meta_pts)
    assert len(gt_pts) == 1
    assert (gt_pts[0].x_m, gt_pts[0].y_m) == (1.0, 1.5)

    # 5. Simulator meta["truth"]["events"] (must exclude chirps)
    meta_sim = {
        "truth": {
            "events": [
                {"t_s": 1.0, "x_m": 4.0, "y_m": 0.4, "kind": "chirp"},
                {"t_s": 14.0, "x_m": 3.0, "y_m": 2.5, "kind": "clap"},
                {"t_s": 16.5, "x_m": 5.5, "y_m": 3.5, "kind": "drone"},
            ]
        }
    }
    gt_sim = load_ground_truth(meta_sim)
    assert len(gt_sim) == 2
    assert [e.kind for e in gt_sim] == ["clap", "drone"]
    assert (gt_sim[0].x_m, gt_sim[0].y_m, gt_sim[0].t_s) == (3.0, 2.5, 14.0)


def test_recording_time_matching_keeps_a_missed_early_trial_unmatched() -> None:
    session = SimpleNamespace(
        meta={}, slices={"p1": SimpleNamespace(start_sample_index=480000, sample_rate=48000)},
    )
    engine = SimpleNamespace(ref_phone_id="p1", to_local=lambda _pid, t: t * 48000)
    ground_truth = [GroundTruthEvent(x, 0.0, t_s=t) for x, t in [(1.0, 1.0), (4.0, 3.0), (9.0, 3.4)]]
    detections = [(13.0, 4.0, 0.0), (13.19, 9.0, 0.0)]

    assert _match_errors(detections, ground_truth, session, engine) == [0.0, 0.0]
    untimed = [GroundTruthEvent(gt.x_m, gt.y_m) for gt in ground_truth]
    assert _match_errors(detections, untimed, session, engine) == []


def test_sweep_caps_grid_at_200(tmp_path: Path) -> None:
    dummy = tmp_path / "unused.npz"
    with pytest.raises(ValueError, match="200"):
        sweep_recording(dummy, grid=[TuneParams()] * 201)

    with pytest.raises(ValueError, match="200"):
        sweep_recording(dummy, grid=[TuneParams()], max_combos=201)

    grid = default_grid()
    assert 1 <= len(grid) <= 72
    assert TuneParams() in grid


def test_format_markdown_handles_all_missed() -> None:
    all_missed = TuneRow(
        params=TuneParams(f_min=2000.0, f_max=3000.0),
        mean_error_m=None,
        median_error_m=None,
        p90_error_m=None,
        missed=10,
        total=10,
        runtime_ms=12.3,
    )
    good = TuneRow(
        params=TuneParams(),
        mean_error_m=0.095,
        median_error_m=0.080,
        p90_error_m=0.140,
        missed=0,
        total=10,
        runtime_ms=15.0,
    )
    md = format_tuning_markdown([all_missed, good], recording_name="test.npz", elapsed_s=1.23)
    assert "mean_m" in md
    assert "median_m" in md
    assert "p90_m" in md
    assert "missed" in md
    assert "1.23" in md

    data_lines = [
        ln
        for ln in md.splitlines()
        if ln.startswith("| ") and not ln.startswith("| rank") and not ln.startswith("| ---")
    ]
    assert len(data_lines) == 2
    # Good row sorts first; all-missed row sorts last with '-' for stats and '10/10' for missed.
    assert "0.095" in data_lines[0]
    assert "0/10" in data_lines[0]
    assert "| - | - | - | 10/10 |" in data_lines[1]


def test_tune_cli_on_sim_recording(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rec = judge_scenario(seed=0)
    npz_path = save_npz(rec, tmp_path / "sim_test.npz")

    small_grid = [
        TuneParams(),
        TuneParams(f_min=500.0, f_max=4000.0, phat_beta=0.6),
    ]
    rows, elapsed_s = sweep_recording(npz_path, grid=small_grid)
    assert elapsed_s > 0.0
    assert len(rows) == 2
    default_row = next(r for r in rows if r.params == TuneParams())
    assert default_row.mean_error_m is not None
    assert default_row.mean_error_m < 0.20
    assert default_row.missed == 0
    assert default_row.total == 10

    truth = json.loads(str(rec["meta"]))["truth"]
    first_phone = truth["phones"][min(truth["phones"])]
    judge_csv = tmp_path / "judge.csv"
    with judge_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial_idx", "gt_x_m", "gt_y_m", "timestamp_s", "t_s"])
        for i, event in enumerate((e for e in truth["events"] if e["kind"] != "chirp"), start=1):
            offset = (event["t_s"] - first_phone["start_s"]) * (1 + first_phone["drift_ppm"] * 1e-6)
            writer.writerow([i, event["x_m"], event["y_m"], 1760000000 + event["t_s"], offset])

    rc = main([str(npz_path), "--truth", str(judge_csv), "--out", str(tmp_path)], grid=small_grid)
    assert rc == 0
    out_md = tmp_path / "tuning-sim_test.md"
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    for col in ("mean_m", "median_m", "p90_m", "missed"):
        assert col in content

    captured = capsys.readouterr().out
    assert "Wall-clock:" in captured
    assert "0/10" in captured
