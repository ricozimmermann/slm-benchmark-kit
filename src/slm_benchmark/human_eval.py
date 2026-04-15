from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class SamplingPlan:
    sample_size: int = 120
    seed: int = 42
    overlap_rate: float = 0.25


def _stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or df.empty:
        return df.head(0).copy()

    rng = np.random.default_rng(seed)
    d = df.copy().reset_index(names="_orig_idx")
    group_cols = ["model", "task_type", "difficulty"]

    grouped = d.groupby(group_cols, dropna=False)
    total = len(d)

    parts = []
    remaining = sample_size
    groups = list(grouped)

    for i, (key, g) in enumerate(groups):
        if i == len(groups) - 1:
            n = max(0, min(len(g), remaining))
        else:
            prop = len(g) / total
            n = int(round(sample_size * prop))
            n = max(1, min(len(g), n))
        remaining -= n

        if n > 0:
            take = g.sample(n=n, random_state=int(rng.integers(0, 10_000_000)))
            parts.append(take)

    if not parts:
        return d.head(0).drop(columns=["_orig_idx"], errors="ignore").reset_index(drop=True)

    sampled = pd.concat(parts, ignore_index=True)

    # If we overshot due to min(1), trim randomly.
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # If we undershot, top up from leftovers.
    if len(sampled) < sample_size:
        missing = sample_size - len(sampled)
        chosen_ids = set(sampled["_orig_idx"].tolist())
        leftovers = d.loc[~d["_orig_idx"].isin(chosen_ids)]
        if len(leftovers) > 0:
            topup = leftovers.sample(n=min(missing, len(leftovers)), random_state=seed + 1)
            sampled = pd.concat([sampled, topup], ignore_index=True)

    return sampled.drop(columns=["_orig_idx"], errors="ignore").reset_index(drop=True)


def prepare_blind_human_eval(
    raw_results_path: str | Path,
    out_assignment_csv: str | Path,
    out_key_csv: str | Path,
    evaluators: Sequence[str],
    plan: SamplingPlan = SamplingPlan(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not evaluators:
        raise ValueError("At least one evaluator must be provided")
    if plan.sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if not 0.0 <= plan.overlap_rate <= 1.0:
        raise ValueError("overlap_rate must be between 0 and 1")

    df = pd.read_json(raw_results_path, lines=True)
    df = df[df["valid_response"] == True].copy()  # noqa: E712

    if df.empty:
        raise ValueError("No valid_response rows found in raw benchmark file")

    sampled = _stratified_sample(df, sample_size=plan.sample_size, seed=plan.seed)
    sampled = sampled.reset_index(drop=True)
    sampled["blind_id"] = [f"BLIND-{i+1:04d}" for i in range(len(sampled))]

    key_cols = [
        "blind_id",
        "model",
        "item_id",
        "task_type",
        "difficulty",
        "temperature",
        "top_p",
        "top_k",
        "repetition",
        "score_aggregated",
        "latency_ms",
    ]
    key_df = sampled[key_cols].copy()

    # Primary assignment split + overlap for agreement.
    rng = np.random.default_rng(plan.seed)
    evaluator_list = list(evaluators)
    base_assign = rng.choice(evaluator_list, size=len(sampled), replace=True)
    sampled["primary_evaluator"] = base_assign

    overlap_n = min(len(sampled), int(round(len(sampled) * plan.overlap_rate)))
    overlap_ids = set(rng.choice(sampled["blind_id"], size=overlap_n, replace=False).tolist())

    rows = []
    for _, row in sampled.iterrows():
        assigned = [row["primary_evaluator"]]
        if row["blind_id"] in overlap_ids and len(evaluator_list) > 1:
            alt = [e for e in evaluator_list if e != row["primary_evaluator"]]
            assigned.append(rng.choice(alt))

        for ev in assigned:
            rows.append(
                {
                    "evaluator_id": ev,
                    "blind_id": row["blind_id"],
                    "task_type": row["task_type"],
                    "difficulty": row["difficulty"],
                    "response_text": row["response_text"],
                    "score_technical": "",
                    "score_completeness": "",
                    "score_actionability": "",
                    "score_clarity": "",
                    "score_format": "",
                    "score_overall": "",
                    "notes": "",
                }
            )

    assignment_df = pd.DataFrame(rows).sort_values(["evaluator_id", "blind_id"]).reset_index(drop=True)

    Path(out_assignment_csv).parent.mkdir(parents=True, exist_ok=True)
    assignment_df.to_csv(out_assignment_csv, index=False)
    key_df.to_csv(out_key_csv, index=False)

    return assignment_df, key_df
