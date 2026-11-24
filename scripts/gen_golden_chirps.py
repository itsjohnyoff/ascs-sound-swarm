"""Regenerate shared/chirp_<fs>.f32 golden vectors (raw little-endian float32)."""

from pathlib import Path

from shared.chirp import generate_lfm_chirp

SHARED = Path(__file__).resolve().parents[1] / "shared"

if __name__ == "__main__":
    for fs in (48000, 44100):
        generate_lfm_chirp(fs).astype("<f4").tofile(SHARED / f"chirp_{fs}.f32")
        print(f"wrote shared/chirp_{fs}.f32")
