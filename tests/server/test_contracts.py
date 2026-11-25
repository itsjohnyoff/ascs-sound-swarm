"""CONTRACTS 1 and 7: golden chirp vectors, v2 constants, facade stub signatures."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from shared import protocol
from shared.chirp import generate_lfm_chirp

SHARED = Path(__file__).resolve().parents[2] / "shared"


@pytest.mark.parametrize("fs", [48000, 44100])
def test_golden_chirp_matches_generator(fs):
    golden = np.fromfile(SHARED / f"chirp_{fs}.f32", dtype="<f4")
    chirp = generate_lfm_chirp(fs)
    assert golden.shape == chirp.shape == (round(fs * protocol.CHIRP_DURATION_S),)
    assert np.max(np.abs(golden - chirp)) < 1e-6


def test_dsp_chirp_reuses_shared_implementation():
    from backend.dsp import chirp as dsp_chirp
    assert dsp_chirp.generate_lfm_chirp is generate_lfm_chirp


def test_v2_constants():
    assert protocol.AUTO_CHIRP_PERIOD_S == 30.0
    assert protocol.CLOCK_FIT_MAX_AGE_S == 600.0
    assert protocol.PLAYOUT_SLACK_S == 0.4
    assert protocol.EVENT_WINDOW_PRE_S == 0.015
    assert protocol.EVENT_WINDOW_POST_S == 0.065
    assert protocol.SUSTAINED_WINDOW_S == 0.75


def test_clock_model_in_shared_protocol():
    from backend.dsp.clock import ClockModel as DspClockModel
    assert hasattr(protocol, "ClockModel")
    assert DspClockModel is protocol.ClockModel


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_stub_signatures_match_contracts():
    from backend.server import stubs
    s = stubs.StubSyncEngine
    assert _params(s.add_chirp_observation) == ["self", "phone_buffers", "phones", "room", "mode"]
    assert _params(s.clock) == ["self", "phone_id"]
    assert _params(s.to_global) == ["self", "phone_id", "sample_index"]
    assert _params(s.to_local) == ["self", "phone_id", "t_global"]
    assert _params(s.mark_clock_jump) == ["self", "phone_id"]
    assert _params(s.extract_aligned_window) == ["self", "phone_buffers", "t_start", "t_end", "target_fs"]
    assert _params(stubs.StubOnsetDetector.push) == ["self", "samples", "start_sample_index", "fs"]
    assert _params(stubs.StubLocalizationPipeline.localize) == ["self", "aligned", "phones", "room", "kind", "fs"]
    assert _params(stubs.StubSoundClassifier.classify_swarm) == ["self", "aligned", "fs"]
    assert stubs.StubSoundClassifier().classify_swarm({}).sound_class == "other"
    assert stubs.StubSyncEngine().to_global("p", 0) is None
