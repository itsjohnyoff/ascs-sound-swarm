"""SoundClassifier (CONTRACTS 7). Heuristic rules now; a trained model can replace `_score` later."""

from __future__ import annotations

import numpy as np

from backend.classifier.features import Features, extract
from shared.protocol import ClassificationResult

TOP_PHONES = 3


def _sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def _score(f: Features) -> dict[str, float]:
    """Soft rule scores in [0, 1]; normalized to probabilities by the caller."""
    drone = _sigmoid((f.comb - 12.0) / 2.0) * _sigmoid((f.active_frac - 0.6) / 0.08) * _sigmoid((f.f0_stability - 0.5) / 0.1)
    clap = (_sigmoid((f.concentration - 0.6) / 0.06) * _sigmoid((f.crest - 4.0) / 0.7)
            * _sigmoid((12.0 - f.comb) / 2.0))
    other = 0.25 * (1 - max(drone, clap))
    return {"drone": drone, "clap": clap, "other": other}


class SoundClassifier:
    backend_name = "heuristic"

    def classify_swarm(self, aligned: dict[str, np.ndarray], fs: int = 48000) -> ClassificationResult:
        chans = [np.asarray(x, dtype=np.float64) for x in aligned.values() if len(x) >= int(0.02 * fs)]
        if not chans:
            return ClassificationResult(sound_class="other", is_drone=False, confidence=0.0,
                                        probs={"drone": 0.0, "clap": 0.0, "other": 1.0})
        # Fuse the 3 loudest phones (highest SNR proxy), weighted by their RMS.
        chans.sort(key=lambda x: -float(np.mean(x * x)))
        feats = [extract(x, fs) for x in chans[:TOP_PHONES]]
        weights = np.array([f.rms for f in feats])
        total = {k: 0.0 for k in ("drone", "clap", "other")}
        for f, w in zip(feats, weights / weights.sum()):
            for k, v in _score(f).items():
                total[k] += w * v
        s = sum(total.values()) or 1.0
        probs = {k: round(v / s, 4) for k, v in total.items()}
        probs["other"] = round(1.0 - probs["drone"] - probs["clap"], 4)
        label = max(probs, key=probs.get)
        return ClassificationResult(sound_class=label, is_drone=label == "drone", confidence=probs[label],
                                    probs=probs, backend_name=self.backend_name)
