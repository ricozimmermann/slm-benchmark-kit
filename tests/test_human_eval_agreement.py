from __future__ import annotations

import json

import pandas as pd

from slm_benchmark.agreement import agreement_markdown_report, pairwise_agreement
from slm_benchmark.human_eval import SamplingPlan, prepare_blind_human_eval


def _write_raw_results(path):
    rows = [
        {
            "model": "m1",
            "item_id": "i1",
            "task_type": "bug_detection",
            "difficulty": "easy",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repetition": 1,
            "score_aggregated": 8.0,
            "latency_ms": 10,
            "valid_response": True,
            "response_text": "resp1",
        },
        {
            "model": "m2",
            "item_id": "i2",
            "task_type": "refactoring",
            "difficulty": "hard",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repetition": 1,
            "score_aggregated": 6.0,
            "latency_ms": 20,
            "valid_response": True,
            "response_text": "resp2",
        },
        {
            "model": "m1",
            "item_id": "i3",
            "task_type": "test_generation",
            "difficulty": "medium",
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 50,
            "repetition": 1,
            "score_aggregated": 7.0,
            "latency_ms": 12,
            "valid_response": True,
            "response_text": "resp3",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_prepare_blind_and_agreement_report(tmp_path):
    raw = tmp_path / "raw.jsonl"
    assignment = tmp_path / "assignment.csv"
    key = tmp_path / "key.csv"
    report = tmp_path / "agreement.md"

    _write_raw_results(raw)

    assignment_df, key_df = prepare_blind_human_eval(
        raw_results_path=raw,
        out_assignment_csv=assignment,
        out_key_csv=key,
        evaluators=["eval01", "eval02", "eval03"],
        plan=SamplingPlan(sample_size=3, seed=1, overlap_rate=1.0),
    )

    assert not assignment_df.empty
    assert not key_df.empty
    assert assignment.exists()
    assert key.exists()

    scored = assignment_df.copy()
    scored["score_overall"] = 7
    scored.to_csv(assignment, index=False)

    pair_df = agreement_markdown_report(assignment, report, key_csv_path=key)

    assert not pair_df.empty
    assert report.exists()


def test_pairwise_agreement_requires_overlap():
    df = pd.DataFrame(
        [
            {"evaluator_id": "eval01", "blind_id": "b1", "score_overall": 7},
            {"evaluator_id": "eval02", "blind_id": "b2", "score_overall": 8},
        ]
    )

    out = pairwise_agreement(df)
    assert len(out) == 1
    assert out.iloc[0]["n_overlap"] == 0


def test_pairwise_agreement_constant_scores_returns_nan_metrics():
    df = pd.DataFrame(
        [
            {"evaluator_id": "eval01", "blind_id": "b1", "score_overall": 7},
            {"evaluator_id": "eval01", "blind_id": "b2", "score_overall": 7},
            {"evaluator_id": "eval01", "blind_id": "b3", "score_overall": 7},
            {"evaluator_id": "eval02", "blind_id": "b1", "score_overall": 7},
            {"evaluator_id": "eval02", "blind_id": "b2", "score_overall": 7},
            {"evaluator_id": "eval02", "blind_id": "b3", "score_overall": 7},
        ]
    )

    out = pairwise_agreement(df)

    assert len(out) == 1
    assert pd.isna(out.iloc[0]["spearman"])
    assert pd.isna(out.iloc[0]["kendall"])
    assert pd.isna(out.iloc[0]["weighted_kappa"])
