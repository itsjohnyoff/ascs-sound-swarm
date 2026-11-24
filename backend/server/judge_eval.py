"""In-memory trial log for the judge evaluation."""

from __future__ import annotations

import csv
import io
import math
from statistics import mean, median
import time

import numpy as np

from shared.protocol import EvalSummary, EvalTrial, LocalizationResult, SoundClass


_CSV_COLUMNS = (
    "trial_idx",
    "label",
    "expected_class",
    "gt_x_m",
    "gt_y_m",
    "est_x_m",
    "est_y_m",
    "error_m",
    "missed",
    "pred_class",
    "class_confidence",
    "residual_ms",
    "timestamp_s",
)


class JudgeEval:
    def __init__(self) -> None:
        self.trials: list[EvalTrial] = []

    def add_trial(
        self,
        label: str,
        gt_x_m: float,
        gt_y_m: float,
        expected_class: SoundClass,
        event: LocalizationResult | None,
    ) -> EvalTrial:
        trial = EvalTrial(
            trial_idx=len(self.trials) + 1,
            label=label,
            gt_x_m=gt_x_m,
            gt_y_m=gt_y_m,
            expected_class=expected_class,
            est_x_m=event.x_m if event else None,
            est_y_m=event.y_m if event else None,
            error_m=math.hypot(event.x_m - gt_x_m, event.y_m - gt_y_m) if event else None,
            missed=event is None,
            pred_class=event.sound_class if event else None,
            class_confidence=event.class_confidence if event else None,
            residual_ms=event.residual_ms if event else None,
            timestamp_s=event.timestamp_s if event else time.time(),
        )
        self.trials.append(trial)
        return trial

    def summary(self) -> EvalSummary:
        errors = [trial.error_m for trial in self.trials if trial.error_m is not None]
        total = len(self.trials)
        detected = len(errors)
        return EvalSummary(
            total=total,
            detected=detected,
            missed=total - detected,
            mean_error_m=mean(errors) if errors else None,
            median_error_m=median(errors) if errors else None,
            p90_error_m=float(np.percentile(errors, 90, method="linear")) if errors else None,
            max_error_m=max(errors) if errors else None,
        )

    def export_csv(self) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(trial.model_dump() for trial in self.trials)
        return output.getvalue()

    def clear(self) -> None:
        self.trials.clear()
