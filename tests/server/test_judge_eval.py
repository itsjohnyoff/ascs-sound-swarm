import csv
import io

import pytest

from backend.server.judge_eval import JudgeEval
from shared.protocol import EvalSummary, LocalizationResult


def make_event(x_m: float, y_m: float) -> LocalizationResult:
    return LocalizationResult(
        event_id="evt-0014",
        timestamp_s=123.0,
        x_m=x_m,
        y_m=y_m,
        uncertainty_m=0.2,
        residual_ms=0.35,
        confidence=0.8,
        phones_used=["a", "b", "c"],
        sound_class="drone",
        class_confidence=0.9,
    )


def test_records_detection_and_miss_in_order():
    judge = JudgeEval()
    detected = judge.add_trial("Pos 1", 1, 2, "clap", make_event(4, 6))
    missed = judge.add_trial("Pos 2", 2, 3, "drone", None)

    assert judge.trials == [detected, missed]
    assert detected.trial_idx == 1
    assert detected.label == "Pos 1"
    assert detected.expected_class == "clap"
    assert (detected.gt_x_m, detected.gt_y_m) == (1, 2)
    assert (detected.est_x_m, detected.est_y_m) == (4, 6)
    assert detected.error_m == 5
    assert detected.missed is False
    assert detected.pred_class == "drone"
    assert detected.class_confidence == 0.9
    assert detected.residual_ms == 0.35
    assert detected.timestamp_s == 123.0
    assert missed.trial_idx == 2
    assert missed.missed is True
    assert missed.error_m is None
    assert missed.est_x_m is None
    assert missed.est_y_m is None
    assert missed.pred_class is None
    assert missed.class_confidence is None
    assert missed.residual_ms is None


def test_summary_uses_linear_percentile_and_ignores_misses():
    judge = JudgeEval()
    judge.add_trial("Pos 1", 0, 0, "clap", make_event(0, 0))
    judge.add_trial("Pos 2", 0, 0, "clap", None)
    judge.add_trial("Pos 3", 0, 0, "clap", make_event(3, 4))
    judge.add_trial("Pos 4", 0, 0, "clap", make_event(9, 0))

    summary = judge.summary()
    assert (summary.total, summary.detected, summary.missed) == (4, 3, 1)
    assert summary.mean_error_m == pytest.approx(14 / 3)
    assert summary.median_error_m == 5
    assert summary.p90_error_m == pytest.approx(8.2)
    assert summary.max_error_m == 9


def test_empty_and_all_missed_summaries_and_clear():
    judge = JudgeEval()
    empty = EvalSummary(total=0, detected=0, missed=0)
    assert judge.summary() == empty

    judge.add_trial("Pos 1", 0, 0, "clap", None)
    assert judge.summary() == EvalSummary(total=1, detected=0, missed=1)
    judge.clear()

    assert judge.trials == []
    assert judge.summary() == empty
    assert len(judge.export_csv().splitlines()) == 1
    assert judge.add_trial("Pos 1", 0, 0, "clap", None).trial_idx == 1


def test_csv_uses_contract_columns_and_quotes_labels():
    judge = JudgeEval()
    label = 'Pos "1", near\r\nthe door'
    judge.add_trial(label, 1, 2, "clap", make_event(4, 6))
    missed = judge.add_trial("Pos 2", 0, 0, "other", None)

    content = judge.export_csv()
    rows = list(csv.reader(io.StringIO(content, newline="")))
    assert content.endswith("\r\n")
    assert rows[0] == [
        "trial_idx", "label", "expected_class", "gt_x_m", "gt_y_m",
        "est_x_m", "est_y_m", "error_m", "missed", "pred_class",
        "class_confidence", "residual_ms", "timestamp_s",
    ]
    assert rows[1] == [
        "1", label, "clap", "1.0", "2.0", "4.0", "6.0", "5.0", "False",
        "drone", "0.9", "0.35", "123.0",
    ]
    assert rows[2] == [
        "2", "Pos 2", "other", "0.0", "0.0", "", "", "", "True", "", "", "",
        str(missed.timestamp_s),
    ]
