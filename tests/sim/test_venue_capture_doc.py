from pathlib import Path

DOC = Path("analysis/venue_capture.md")


def test_venue_capture_protocol_has_required_sections():
    text = DOC.read_text(encoding="utf-8")
    for token in (
        "spike-01.npz",
        "-venue-claps.npz",
        "-venue-noise.npz",
        "-venue-drone.npz",
        "20 s",
        "python -m analysis.spike",
        "python -m analysis.classifier_eval",
        "analysis/LOG.md",
        "Gate 0",
        "macro F1",
    ):
        assert token in text, f"missing required token: {token}"
