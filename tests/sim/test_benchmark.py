"""Agent 3 phase 2: offline benchmark and virtual phone framing; Agent 2 phase 2 MUST."""

import json

import numpy as np
import pytest

from shared.protocol import decode_audio_frame
from sim.benchmark import format_markdown, run_benchmark
from sim.room import judge_scenario
from sim.virtual_phones import frames_for


def test_benchmark_on_clean_judge_scenario():
    result = run_benchmark(judge_scenario(seed=3))
    assert len(result.rows) == 10
    assert result.missed == 0
    assert result.mean_error_m < 0.3
    assert [r.sound_class for r in result.rows] == [r.expected for r in result.rows]
    md = format_markdown(result)
    assert "| gt |" in md and "mean" in md and "inside_hull" in md


def test_benchmark_with_one_occluded_phone_stays_under_0_3_m_inside_hull():
    # p2's direct path is blocked: weak direct, strong reflection 2.5 ms later.
    rec = judge_scenario(seed=5, occluded=("p2", 0.15, 0.0025, 1.0))
    result = run_benchmark(rec)
    inside = [r for r in result.rows if r.inside_hull and not r.missed]
    errors = [r.error_m for r in inside]
    print(format_markdown(result))
    assert len(inside) >= 7
    assert float(np.mean(errors)) < 0.3


def test_benchmark_with_stubs_reports_everything_missed():
    result = run_benchmark(judge_scenario(seed=3), stubs=True)
    assert result.missed == 10 and result.mean_error_m is None


def test_virtual_phone_frames_are_contiguous_2048_sample_frames():
    rec = judge_scenario(seed=3)
    meta = json.loads(str(rec["meta"]))
    frames = list(frames_for(rec, "p1"))
    decoded = [decode_audio_frame(f.payload) for f in frames]
    start = meta["truth"]["phones"]["p1"]["start_index"]
    assert decoded[0].sample_index == start
    assert all(d.num_samples == 2048 for d in decoded[:-1])
    assert all(b.sample_index == a.sample_index + a.num_samples for a, b in zip(decoded, decoded[1:]))
    assert [d.seq for d in decoded] == list(range(len(decoded)))
    assert frames[1].at_s - frames[0].at_s == pytest.approx(2048 / meta["truth"]["phones"]["p1"]["fs"], rel=1e-3)
