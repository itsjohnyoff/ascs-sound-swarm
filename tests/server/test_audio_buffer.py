import struct

import numpy as np
import pytest

from backend.server.audio_buffer import PhoneRingBuffer, decode_audio_frame
from shared.protocol import (
    FLAG_CLIPPING,
    FLAG_CLOCK_JUMP,
    decode_audio_frame as shared_decode_audio_frame,
    encode_audio_frame,
)


SAMPLE_RATE = 8000


def make_frame(seq, sample_index, samples, flags=0, sample_rate=SAMPLE_RATE):
    payload = encode_audio_frame(
        np.asarray(samples, dtype=np.float32), seq, sample_rate, sample_index, flags=flags
    )
    return decode_audio_frame(payload)


def test_wire_header_and_pcm_round_trip():
    pcm = np.array([-32768, -1, 0, 32767], dtype="<i2")
    sample_index = 2**40 + 13
    header = struct.pack("<4sBBHIIQd", b"ASCS", 1, FLAG_CLIPPING, 4, 17, 48000, sample_index, 12.5)

    assert len(header) == 32
    assert decode_audio_frame is shared_decode_audio_frame
    frame = decode_audio_frame(header + pcm.tobytes())

    assert frame.flags == FLAG_CLIPPING
    assert frame.num_samples == pcm.size
    assert frame.seq == 17
    assert frame.sample_rate == 48000
    assert frame.sample_index == sample_index
    assert frame.client_time_ms == 12.5
    assert frame.samples_f32.dtype == np.float32
    np.testing.assert_array_equal(frame.samples_f32, pcm.astype(np.float32) / 32768)


@pytest.mark.parametrize(
    ("edit", "error"),
    [
        (lambda payload: payload[:31], "too short"),
        (lambda payload: b"NOPE" + payload[4:], "magic"),
        (lambda payload: payload[:4] + b"\x02" + payload[5:], "version"),
        (lambda payload: payload[:6] + b"\x00\x00" + payload[8:], "Empty PCM"),
        (lambda payload: payload[:12] + struct.pack("<I", 7999) + payload[16:], "sample_rate"),
        (lambda payload: payload[:12] + struct.pack("<I", 192001) + payload[16:], "sample_rate"),
        (lambda payload: payload[:-1], "length mismatch"),
        (lambda payload: payload + b"\x00", "length mismatch"),
    ],
)
def test_rejects_malformed_wire_frames(edit, error):
    payload = encode_audio_frame(np.zeros(2, dtype=np.float32), 17, 48000, 100)
    with pytest.raises(ValueError, match=error):
        decode_audio_frame(edit(payload))


def test_empty_slice_reports_buffer_identity():
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE)

    recent = buffer.get_recent_slice(1)

    assert buffer.latest_sample_index == 0
    assert buffer.dropped_frames == buffer.clipping_count == 0
    assert buffer.rms == 0
    assert recent.phone_id == "phone-1"
    assert recent.sample_rate == SAMPLE_RATE
    assert recent.start_sample_index == 0
    assert recent.samples_f32.size == 0
    assert recent.samples_f32.dtype == np.float32


def test_ring_wrap_preserves_exact_indices_and_snapshot():
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE)
    start = 2**40 + 7
    source = np.linspace(-0.5, 0.5, 100_000, dtype=np.float32)
    first = make_frame(0, start, source[:50_000])
    second = make_frame(1, start + 50_000, source[50_000:])
    assert buffer.append(first)
    assert buffer.append(second)
    expected = np.concatenate((first.samples_f32, second.samples_f32))

    all_recent = buffer.get_recent_slice(20)
    assert buffer.latest_sample_index == start + 100_000
    assert all_recent.start_sample_index == start + 4000
    assert all_recent.samples_f32.flags.c_contiguous
    np.testing.assert_array_equal(all_recent.samples_f32, expected[-96_000:])

    one_second = buffer.get_recent_slice(1)
    assert one_second.start_sample_index == start + 92_000
    np.testing.assert_array_equal(one_second.samples_f32, expected[-SAMPLE_RATE:])

    assert buffer.append(make_frame(2, start + 100_000, np.zeros(8000)))
    np.testing.assert_array_equal(all_recent.samples_f32, expected[-96_000:])


def test_drops_count_once_and_snapshot_contains_only_contiguous_suffix():
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE)
    assert buffer.append(make_frame(12, 100, [0.125] * 4))

    first_after_gap = make_frame(15, 112, [0.25] * 4)
    assert buffer.append(first_after_gap)
    assert buffer.dropped_frames == 2

    second_after_gap = make_frame(16, 123, [0.5, 0.25])
    next_frame = make_frame(17, 125, [0.125])
    assert buffer.append(second_after_gap)
    assert buffer.append(next_frame)

    recent = buffer.get_recent_slice(1)
    assert buffer.dropped_frames == 3
    assert recent.start_sample_index == 123
    assert buffer.latest_sample_index == 126
    np.testing.assert_array_equal(
        recent.samples_f32, np.concatenate((second_after_gap.samples_f32, next_frame.samples_f32))
    )


def test_sequence_gap_with_contiguous_samples_keeps_both_frames():
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE)
    first = make_frame(1, 100, [0.125] * 4)
    second = make_frame(3, 104, [0.25] * 4)
    assert buffer.append(first)
    assert buffer.append(second)

    recent = buffer.get_recent_slice(1)
    assert buffer.dropped_frames == 1
    assert recent.start_sample_index == 100
    np.testing.assert_array_equal(
        recent.samples_f32, np.concatenate((first.samples_f32, second.samples_f32))
    )


def test_rejects_duplicate_stale_overlapping_and_wrong_rate_frames():
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE)
    valid = make_frame(8, 100, [0.25] * 4)
    assert buffer.append(valid)
    assert not buffer.append(valid)
    assert not buffer.append(make_frame(7, 80, [-1.0] * 4))
    assert not buffer.append(make_frame(7, 104, [-1.0] * 4))
    assert not buffer.append(make_frame(9, 103, [-1.0] * 4))
    assert not buffer.append(make_frame(9, 104, [-1.0] * 4, sample_rate=48000))

    recent = buffer.get_recent_slice(1)
    assert recent.start_sample_index == 100
    assert buffer.latest_sample_index == 104
    assert buffer.dropped_frames == buffer.clipping_count == 0
    np.testing.assert_array_equal(recent.samples_f32, valid.samples_f32)


def test_sequence_wrap_keeps_audio_contiguous():
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE)
    for offset, seq in enumerate([0xFFFFFFFE, 0xFFFFFFFF, 0, 1]):
        assert buffer.append(make_frame(seq, 100 + offset, [0.125]))

    recent = buffer.get_recent_slice(1)
    assert buffer.dropped_frames == 0
    assert recent.start_sample_index == 100
    assert buffer.latest_sample_index == 104
    assert recent.samples_f32.size == 4


def test_clock_jump_resets_audio_and_discontinuity_counter():
    clock_jumps = []
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE, on_clock_jump=clock_jumps.append)
    assert buffer.append(make_frame(5, 100, [0.25] * 4))
    assert buffer.append(make_frame(7, 108, [0.25] * 4))
    assert buffer.dropped_frames == 1

    reset = make_frame(0, 3, [0.125, 0.5], flags=FLAG_CLOCK_JUMP)
    assert buffer.append(reset)

    recent = buffer.get_recent_slice(1)
    assert clock_jumps == ["phone-1"]
    assert buffer.dropped_frames == 0
    assert recent.start_sample_index == 3
    assert buffer.latest_sample_index == 5
    np.testing.assert_array_equal(recent.samples_f32, reset.samples_f32)

    assert buffer.append(make_frame(1, 5, [0.125]))
    assert buffer.dropped_frames == 0


def test_telemetry_uses_last_frame_rms_and_counts_clipped_frames():
    buffer = PhoneRingBuffer("phone-1", SAMPLE_RATE)
    clipped = make_frame(0, 0, [-1, 1, 0])
    assert buffer.append(clipped)
    assert buffer.clipping_count == 1
    assert buffer.rms == pytest.approx(np.sqrt(np.mean(clipped.samples_f32 ** 2)))

    quiet = make_frame(1, 3, [0.125, -0.25])
    assert buffer.append(quiet)
    assert buffer.clipping_count == 1
    assert buffer.rms == pytest.approx(np.sqrt(np.mean(quiet.samples_f32 ** 2)))
