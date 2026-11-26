"""SoundClassifier (CONTRACTS 7).

Loads the trained model `models/drone_classifier.joblib` when it is present and valid,
otherwise falls back to the heuristic rules. Either way, the 3 loudest phones are
classified separately and fused, weighted by RMS (an SNR proxy).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from backend.classifier.features import Features, extract, vector
from shared.protocol import ClassificationResult

log = logging.getLogger(__name__)

TOP_PHONES = 3
CLASSES = ("drone", "clap", "other")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "drone_classifier.joblib"


def _sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def _score(f: Features) -> dict[str, float]:
    """Soft rule scores in [0, 1]; normalized to probabilities by the caller."""
    drone = _sigmoid((f.comb - 12.0) / 2.0) * _sigmoid((f.active_frac - 0.6) / 0.08) * _sigmoid((f.f0_stability - 0.5) / 0.1)
    clap = (_sigmoid((f.concentration - 0.6) / 0.06) * _sigmoid((f.crest - 4.0) / 0.7)
            * _sigmoid((12.0 - f.comb) / 2.0))
    other = 0.25 * (1 - max(drone, clap))
    return {"drone": drone, "clap": clap, "other": other}


def load_model(path: str | Path) -> dict | None:
    """The saved bundle, or None if missing or unusable (the caller falls back to heuristics)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        import joblib

        bundle = joblib.load(path)
        model, names = bundle["model"], tuple(bundle["features"])
        vector(Features(*([0.0] * 13)), names)  # every saved feature name still exists
        if set(model.classes_) - set(CLASSES):
            raise ValueError(f"unexpected classes {model.classes_}")
        return bundle
    except Exception as exc:  # noqa: BLE001 - a broken model file must never take down the server
        log.warning("Ignoring classifier model %s: %s", path, exc)
        return None


class SoundClassifier:
    backend_name = "heuristic"

    def __init__(self, model_path: str | Path | None = DEFAULT_MODEL_PATH) -> None:
        """model_path=None forces the heuristic rules."""
        self._bundle = load_model(model_path) if model_path is not None else None
        if self._bundle is not None:
            self.backend_name = str(self._bundle.get("backend_name", "hgb"))

    def _probs(self, feats: list[Features]) -> list[dict[str, float]]:
        if self._bundle is None:
            return [_score(f) for f in feats]  # raw rule scores; normalized after fusion
        names = tuple(self._bundle["features"])
        model = self._bundle["model"]
        # Scoring 3 rows is ~1 ms of work; letting HGB spin up its OpenMP pool across every
        # core costs ~30 ms per call (and more under load), so keep prediction on one thread.
        with threadpool_limits(limits=1, user_api="openmp"):
            p = model.predict_proba(np.vstack([vector(f, names) for f in feats]))
        return [{k: float(row[list(model.classes_).index(k)]) if k in model.classes_ else 0.0 for k in CLASSES}
                for row in p]

    def classify_swarm(self, aligned: dict[str, np.ndarray], fs: int = 48000) -> ClassificationResult:
        chans = [np.asarray(x, dtype=np.float64) for x in aligned.values() if len(x) >= int(0.02 * fs)]
        if not chans:
            return ClassificationResult(sound_class="other", is_drone=False, confidence=0.0,
                                        probs={"drone": 0.0, "clap": 0.0, "other": 1.0},
                                        backend_name=self.backend_name)
        # Fuse the 3 loudest phones (highest SNR proxy), weighted by their RMS.
        chans.sort(key=lambda x: -float(np.mean(x * x)))
        feats = [extract(x, fs) for x in chans[:TOP_PHONES]]
        weights = np.array([f.rms for f in feats])
        total = {k: 0.0 for k in CLASSES}
        for p, w in zip(self._probs(feats), weights / weights.sum()):
            for k, v in p.items():
                total[k] += w * v
        s = sum(total.values()) or 1.0
        probs = {k: round(v / s, 4) for k, v in total.items()}
        probs["other"] = round(1.0 - probs["drone"] - probs["clap"], 4)
        label = max(probs, key=probs.get)
        return ClassificationResult(sound_class=label, is_drone=label == "drone", confidence=probs[label],
                                    probs=probs, backend_name=self.backend_name)
