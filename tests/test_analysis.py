from __future__ import annotations

from pathlib import Path

import pandas as pd

from slm_benchmark.analysis import generate_markdown_report, summarize_by_model, welch_between_models


def _sample_df():
    return pd.DataFrame(
        [
            {
                "model": "m1",
                "score_aggregated": 8.0,
                "error": "",
                "judge_valid_count": 1,
                "latency_ms": 10,
                "valid_response": True,
                "task_type": "bug_detection",
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "item_id": "i1",
                "judge_rationales": {"j1": "ok", "j2": "judge_parse_fallback_used: x"},
                "judge_scores": {"j1": 8.0, "j2": 7.0},
            },
            {
                "model": "m2",
                "score_aggregated": 6.0,
                "error": "timeout",
                "judge_valid_count": 0,
                "latency_ms": 20,
                "valid_response": True,
                "task_type": "bug_detection",
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "item_id": "i2",
                "judge_rationales": {"j1": "judge_error: err", "j2": "judge_parse_error: err"},
                "judge_scores": {"j1": 6.0, "j2": 6.5},
            },
            {
                "model": "m1",
                "score_aggregated": 7.0,
                "error": "",
                "judge_valid_count": 1,
                "latency_ms": 11,
                "valid_response": True,
                "task_type": "refactoring",
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 50,
                "item_id": "i3",
                "judge_rationales": {"j1": "ok", "j2": "ok"},
                "judge_scores": {"j1": 7.0, "j2": 7.0},
            },
            {
                "model": "m2",
                "score_aggregated": 5.5,
                "error": "",
                "judge_valid_count": 1,
                "latency_ms": 22,
                "valid_response": True,
                "task_type": "refactoring",
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 50,
                "item_id": "i4",
                "judge_rationales": {"j1": "ok", "j2": "ok"},
                "judge_scores": {"j1": 5.0, "j2": 6.0},
            },
        ]
    )


def test_summarize_by_model_not_empty():
    summary = summarize_by_model(_sample_df())

    assert not summary.empty
    assert set(summary["model"]) == {"m1", "m2"}


def test_summarize_by_model_without_judge_valid_count_column():
    df = _sample_df().drop(columns=["judge_valid_count"])

    summary = summarize_by_model(df)

    assert not summary.empty
    assert "judge_all_failed_rate" in summary.columns


def test_welch_between_models_returns_stats_for_two_models():
    out = welch_between_models(_sample_df())

    assert out["model_a"] in {"m1", "m2"}
    assert "p_value" in out


def test_generate_markdown_report_writes_sections(tmp_path):
    out_md = tmp_path / "report.md"
    generate_markdown_report(_sample_df(), out_md)

    text = Path(out_md).read_text(encoding="utf-8")
    assert "# SLM Benchmark Report" in text
    assert "## Model Summary" in text
    assert "## Judge Health" in text
