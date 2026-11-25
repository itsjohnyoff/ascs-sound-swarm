"""Tests for single-event diagnostic SVG plot generation (backend/dsp/diagnostics.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.dsp.diagnostics import EventDiagnostics, generate_event_diagnostics
from sim.room import judge_scenario, save_npz


def _assert_valid_svg(path: Path) -> None:
    assert path.exists(), f"Missing SVG plot: {path}"
    assert path.suffix == ".svg", f"Expected .svg suffix, got {path}"
    assert path.stat().st_size > 500, f"SVG too small: {path.stat().st_size} bytes"
    text = path.read_text(encoding="utf-8").strip()
    assert text.startswith("<svg"), f"SVG does not start with <svg: {text[:40]!r}"
    assert "</svg>" in text, f"SVG missing </svg> closing tag: {path}"


def test_diagnostics_generates_all_three_plots_on_sim_recording(tmp_path: Path) -> None:
    npz_path = save_npz(judge_scenario(seed=0), tmp_path / "diag_sim.npz")
    out_dir = tmp_path / "plots"

    diag = generate_event_diagnostics(npz_path, event_index=0, out_dir=out_dir)

    assert isinstance(diag, EventDiagnostics)
    assert diag.event_index == 0
    assert diag.est_xy is not None
    assert set(diag.plot_paths.keys()) == {"chirp_envelope", "gcc_pairs", "triangle_residuals"}
    for path in diag.plot_paths.values():
        _assert_valid_svg(path)

    assert len(diag.pair_tdoas_ms) == 6
    assert len(diag.triangle_residuals_ms) == 4
    assert max(diag.triangle_residuals_ms.values()) < 1.0


def test_diagnostics_two_phone_recording_does_not_crash(tmp_path: Path) -> None:
    npz_path = save_npz(judge_scenario(n_phones=2, seed=0), tmp_path / "diag_two.npz")
    out_dir = tmp_path / "plots_two"

    diag = generate_event_diagnostics(npz_path, event_index=0, out_dir=out_dir)

    assert diag.triangle_residuals_ms == {}
    assert len(diag.pair_tdoas_ms) == 1
    assert set(diag.plot_paths.keys()) == {"chirp_envelope", "gcc_pairs", "triangle_residuals"}
    for path in diag.plot_paths.values():
        _assert_valid_svg(path)
    tri_svg = diag.plot_paths["triangle_residuals"].read_text(encoding="utf-8")
    assert "Requires" in tri_svg and "3 phones" in tri_svg


def test_diagnostics_invalid_event_index_raises(tmp_path: Path) -> None:
    npz_path = save_npz(judge_scenario(seed=0), tmp_path / "diag_oob.npz")

    with pytest.raises(IndexError):
        generate_event_diagnostics(npz_path, event_index=999, out_dir=tmp_path / "plots_oob")

    with pytest.raises(IndexError):
        generate_event_diagnostics(npz_path, event_index=-1, out_dir=tmp_path / "plots_oob")
