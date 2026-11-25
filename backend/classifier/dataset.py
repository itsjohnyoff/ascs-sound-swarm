"""Training windows for the drone / clap / other model.

Three sources, each window tagged with a *group* (the source recording) so the
train/test split never puts two windows of one recording on both sides:

* synthetic: `sim.sources.TRAINING_SOURCES` rendered through random RIRs, phone-speaker
  playback and noise (always available, no network).
* DroneAudioSet (HF `ahlab-drone-project/DroneAudioSet`, config `drone-only`): a few
  recordings streamed, cut into windows. Needs the `ml` extra and network.
* local: `data/train/<drone|clap|other|noise>/*.wav` (venue recordings). `noise` files are
  used as background for the other classes and as `other` windows themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from backend.classifier.features import extract, vector
from sim.augment import augment, place_in_window
from sim.sources import SRC_FS, TRAINING_SOURCES

log = logging.getLogger(__name__)

CLASSES = ("drone", "clap", "other")
FS = SRC_FS
DRONEAUDIOSET = "ahlab-drone-project/DroneAudioSet"

# Which window kinds each source shows up in, as P(transient).
_P_TRANSIENT = {"drone": 0.1, "multirotor_drone": 0.1, "clap": 1.0, "speech_like": 0.4, "knock": 1.0,
                "noise_burst": 0.1, "whistle": 0.15, "music_like": 0.4}


@dataclass
class Dataset:
    X: list[np.ndarray] = field(default_factory=list)
    y: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    origin: list[str] = field(default_factory=list)  # synthetic | droneaudioset | local

    def add(self, window: np.ndarray, label: str, group: str, origin: str) -> None:
        self.X.append(vector(extract(window, FS)))
        self.y.append(label)
        self.groups.append(group)
        self.origin.append(origin)

    def extend(self, other: "Dataset") -> None:
        for a in ("X", "y", "groups", "origin"):
            getattr(self, a).extend(getattr(other, a))

    def arrays(self):
        return np.vstack(self.X), np.array(self.y), np.array(self.groups), np.array(self.origin)

    def __len__(self) -> int:
        return len(self.y)


def _resample(x: np.ndarray, fs: int) -> np.ndarray:
    if fs == FS:
        return np.asarray(x, dtype=np.float64)
    g = np.gcd(int(fs), FS)
    return resample_poly(np.asarray(x, dtype=np.float64), FS // g, int(fs) // g)


def _kind(rng: np.random.Generator, p_transient: float) -> str:
    return "transient" if rng.random() < p_transient else "sustained"


def synthetic(n_per_class: int, rng: np.random.Generator, *, windows_per_source: int = 2,
              noise_refs: list[np.ndarray] | None = None) -> Dataset:
    ds = Dataset()
    for label, fns in TRAINING_SOURCES.items():
        for i in range(int(np.ceil(n_per_class / windows_per_source))):
            fn = fns[i % len(fns)]
            dur = {"clap": rng.uniform(0.08, 0.2), "knock": rng.uniform(0.08, 0.2)}.get(fn.__name__, rng.uniform(1.0, 1.4))
            src = fn(rng, duration_s=dur)
            for _ in range(windows_per_source):
                y, snr = augment(src, rng, p_speaker=0.35 if label == "drone" else 0.15)
                noise = noise_refs[int(rng.integers(len(noise_refs)))] if noise_refs else None
                w = place_in_window(y, _kind(rng, _P_TRANSIENT[fn.__name__]), rng, snr, noise_ref=noise)
                ds.add(w, label, f"syn-{fn.__name__}-{i}", "synthetic")
    return ds


def _windows_from_long(x: np.ndarray, label: str, group: str, origin: str, n: int, rng: np.random.Generator,
                       *, p_transient: float, p_rir: float, p_speaker: float, snr_db=(10.0, 40.0),
                       noise_refs: list[np.ndarray] | None = None) -> Dataset:
    ds = Dataset()
    seg_len = int(1.2 * FS)
    if len(x) < seg_len:
        return ds
    for _ in range(n):
        i = int(rng.integers(0, len(x) - seg_len + 1))
        y, snr = augment(x[i:i + seg_len], rng, p_rir=p_rir, p_speaker=p_speaker, snr_db=snr_db)
        noise = noise_refs[int(rng.integers(len(noise_refs)))] if noise_refs else None
        ds.add(place_in_window(y, _kind(rng, p_transient), rng, snr, noise_ref=noise), label, group, origin)
    return ds


def droneaudioset(n_recordings: int, windows_per_recording: int, rng: np.random.Generator,
                  *, splits: tuple[str, ...] | None = None) -> Dataset:
    """Stream `n_recordings` drone-only recordings (one per split, spread over the set)."""
    from datasets import load_dataset  # optional `ml` extra

    ds = Dataset()
    splits = splits or tuple(f"train_{k:03d}" for k in np.linspace(1, 28, n_recordings).astype(int))
    for split in splits[:n_recordings]:
        stream = load_dataset(DRONEAUDIOSET, "drone-only", split=split, streaming=True)
        row = next(iter(stream))
        audio = np.asarray(row["audio"]["array"], dtype=np.float64)
        fs = int(row["audio"]["sampling_rate"])
        if audio.ndim == 2:  # (channels, samples) or (samples, channels): take one mic
            audio = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
        audio = _resample(audio[: int(60 * fs)], fs)  # shortcut: first 60 s per recording is plenty of windows
        group = f"das-{row.get('file_path') or split}"
        log.info("DroneAudioSet %s: %s, %.0f s at %d Hz", split, group, len(audio) / FS, fs)
        ds.extend(_windows_from_long(audio, "drone", group, "droneaudioset", windows_per_recording, rng,
                                     p_transient=0.1, p_rir=0.6, p_speaker=0.4))
    return ds


def _read_wav(path: Path) -> np.ndarray:
    import soundfile as sf  # optional `ml` extra

    x, fs = sf.read(str(path), always_2d=True)
    return _resample(x[:, 0], fs)


def _onsets(x: np.ndarray, min_gap_s: float = 0.3, rise_db: float = 15.0) -> list[int]:
    hop = int(0.005 * FS)
    e = np.sqrt(np.mean(x[: len(x) // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-9
    floor = np.median(e)
    out, last = [], -10**9
    for k in np.flatnonzero(20 * np.log10(e / floor) > rise_db):
        if (k - last) * hop >= min_gap_s * FS:
            out.append(int(k * hop))
            last = k
    return out


def local(root: str | Path, rng: np.random.Generator, windows_per_file: int = 40) -> tuple[Dataset, list[np.ndarray]]:
    root = Path(root)
    ds, noise_refs = Dataset(), []
    if not root.exists():
        return ds, noise_refs
    for path in sorted((root / "noise").glob("*.wav")):
        x = _read_wav(path)
        noise_refs.append(x)
        ds.extend(_windows_from_long(x, "other", f"local-{path.name}", "local", windows_per_file, rng,
                                     p_transient=0.3, p_rir=0.0, p_speaker=0.0, snr_db=(60.0, 60.0)))
    for label in CLASSES:
        for path in sorted((root / label).glob("*.wav")):
            x = _read_wav(path)
            group = f"local-{path.name}"
            if label == "clap":  # a file of many claps: one transient window per onset
                for on in _onsets(x):
                    seg = x[max(0, on - int(0.004 * FS)): on + int(0.2 * FS)]
                    for _ in range(3):
                        y, snr = augment(seg, rng, p_rir=0.3, p_speaker=0.0, snr_db=(20.0, 50.0))
                        ds.add(place_in_window(y, "transient", rng, snr), label, group, "local")
            else:
                ds.extend(_windows_from_long(x, label, group, "local", windows_per_file, rng,
                                             p_transient=0.1, p_rir=0.3, p_speaker=0.0, snr_db=(25.0, 60.0)))
    return ds, noise_refs
