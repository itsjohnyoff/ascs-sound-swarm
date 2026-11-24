import asyncio
import threading

import numpy as np
import pytest

from backend.server.processor import Processor
from backend.server.session import Session, compute_gdop_warning
from shared.protocol import ClassificationResult, LocalizationResult, PhoneConfig


class SampleClock:
    def mark_clock_jump(self, phone_id):
        pass

    def is_synced(self, phone_id):
        return True

    def sample_to_global_time(self, phone_id, index):
        return index / 48000

    def extract_aligned_window(self, buffers, start, end, target_fs):
        aligned = {}
        for phone_id, audio in buffers.items():
            begin = round(start * target_fs) - audio.start_sample_index
            finish = round(end * target_fs) - audio.start_sample_index
            aligned[phone_id] = audio.samples_f32[begin:finish]
        return aligned


class Localizer:
    def __init__(self):
        self.calls = []

    def detect_and_localize(self, aligned, phones, room, fs, timestamp_s):
        self.calls.append((threading.get_ident(), len(next(iter(aligned.values())))))
        if not any(np.any(samples) for samples in aligned.values()):
            return None
        return LocalizationResult(
            event_id=f"evt-{len(self.calls)}", timestamp_s=timestamp_s, x_m=1,
            y_m=2, uncertainty_m=0.2, residual_ms=0.1, confidence=0.8,
            phones_used=list(aligned),
        )


class Classifier:
    backend_name = "heuristic"

    def __init__(self, sound_class):
        self.sound_class = sound_class
        self.thread_id = None

    def classify_swarm(self, aligned, fs):
        self.thread_id = threading.get_ident()
        return ClassificationResult(
            sound_class=self.sound_class, is_drone=self.sound_class == "drone",
            confidence=0.9, probs={self.sound_class: 0.9},
        )


def feed(session, seq, start, end, spike=None, amplitude=0):
    from shared.protocol import DecodedAudioFrame

    samples = np.full(end - start, amplitude, dtype=np.float32)
    if spike is not None:
        samples[spike - start] = 0.9
    for pid in session.buffers:
        session.buffers[pid].append(
            DecodedAudioFrame(0, len(samples), seq, 48000, start, 0, samples),
        )


def ready_session(sound_class):
    session = Session(sync_engine=SampleClock(), localizer=Localizer(), classifier=Classifier(sound_class))
    for pid in ("nw", "ne", "sw"):
        session.connect(pid)
        session.phones[pid] = session.phones[pid].model_copy(update={"sync_state": "synced"})
    return session


def test_clap_is_emitted_once_and_a_second_clap_is_retained():
    async def run():
        session = ready_session("clap")
        sent = []
        processor = Processor(session, sent.append)
        feed(session, 0, 0, 48000, spike=40000)
        await processor.process_once()
        assert [event["sound_class"] for event in sent] == ["clap"]
        assert session.localizer.calls[0][1] == 42239
        assert session.localizer.calls[0][0] != threading.get_ident()
        assert session.classifier.thread_id != threading.get_ident()

        feed(session, 1, 48000, 64000)
        await processor.process_once()
        assert len(sent) == 1
        feed(session, 2, 64000, 80000, spike=68000)
        await processor.process_once()
        assert len(sent) == 2
        assert session.detections[sent[-1]["event_id"]].class_confidence == 0.9

    asyncio.run(run())


@pytest.mark.parametrize("diagnostic, count", [(False, 0), (True, 2)])
def test_two_phone_diagnostic_and_drone_updates(diagnostic, count):
    async def run():
        session = ready_session("drone")
        session.disconnect("sw")
        sent = []
        processor = Processor(session, sent.append, min_phones=2 if diagnostic else 3)
        feed(session, 0, 0, 48000, amplitude=0.1)
        await processor.process_once()
        feed(session, 1, 48000, 52800, amplitude=0.1)
        await processor.process_once()
        feed(session, 2, 52800, 62400, amplitude=0.1)
        await processor.process_once()
        await processor.process_once()  # same audio must never republish
        assert [event["sound_class"] for event in sent] == ["drone"] * count

    asyncio.run(run())


def test_geometry_uses_connected_phones_only():
    phones = [PhoneConfig(phone_id=str(x), x_m=x, y_m=x) for x in range(3)]
    assert "collinear" in compute_gdop_warning(phones)
    phones[2].y_m = 0
    assert compute_gdop_warning(phones) is None
    phones[2].connected = False
    assert "at least 3" in compute_gdop_warning(phones)


def test_stalled_phone_cannot_freeze_a_healthy_quorum():
    async def run():
        session = ready_session("drone")
        session.connect("se")
        session.phones["se"] = session.phones["se"].model_copy(update={"sync_state": "synced"})
        sent = []
        processor = Processor(session, sent.append)
        feed(session, 0, 0, 48000, amplitude=0.1)
        await processor.process_once()

        stalled = session.buffers.pop("se")
        feed(session, 1, 48000, 68000, amplitude=0.1)
        session.buffers["se"] = stalled
        await processor.process_once()
        assert len(sent) == 2
        assert "se" not in sent[-1]["phones_used"]

    asyncio.run(run())


def test_new_phone_does_not_reemit_a_buffered_clap():
    async def run():
        session = ready_session("clap")
        sent = []
        processor = Processor(session, sent.append)
        feed(session, 0, 0, 48000, spike=40000)
        await processor.process_once()
        session.connect("se")
        await processor.process_once()
        assert len(sent) == 1

    asyncio.run(run())


def test_analysis_starts_after_the_calibration_capture():
    async def run():
        session = ready_session("clap")
        sent = []
        processor = Processor(session, sent.append)
        feed(session, 0, 0, 48000, spike=35000)
        session.calibration_ends = {pid: 48000 for pid in session.buffers}
        await processor.process_once()
        assert sent == []
        assert session.localizer.calls == []

        feed(session, 1, 48000, 72000, spike=55000)
        await processor.process_once()
        assert len(sent) == 1

    asyncio.run(run())
