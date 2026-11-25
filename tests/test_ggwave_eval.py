import pytest

ggwave = pytest.importorskip("ggwave")


def test_lfm_beats_ggwave_timing():
    from analysis.ggwave_eval import evaluate

    r = evaluate(trials=2, snr_db=20.0)
    assert r["decode_rate"] == 1.0
    assert r["lfm_tdoa_rms_ms"] < 0.05 < r["ggwave_tdoa_rms_ms"]
