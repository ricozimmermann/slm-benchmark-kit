from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.inter_rater import cohens_kappa


def _normalize_scores(df: pd.DataFrame, score_col: str = "score_overall") -> pd.DataFrame:
    d = df.copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=["evaluator_id", "blind_id", score_col])
    d[score_col] = d[score_col].clip(0, 10)
    return d


def pairwise_agreement(df_scored: pd.DataFrame, score_col: str = "score_overall") -> pd.DataFrame:
    d = _normalize_scores(df_scored, score_col=score_col)
    evaluators = sorted(d["evaluator_id"].unique().tolist())

    rows = []
    for a, b in combinations(evaluators, 2):
        da = d[d["evaluator_id"] == a][["blind_id", score_col]].rename(columns={score_col: "score_a"})
        db = d[d["evaluator_id"] == b][["blind_id", score_col]].rename(columns={score_col: "score_b"})
        m = da.merge(db, on="blind_id", how="inner")
        n = len(m)
        if n < 3:
            rows.append({"eval_a": a, "eval_b": b, "n_overlap": n, "spearman": np.nan, "kendall": np.nan, "weighted_kappa": np.nan})
            continue

        spearman = stats.spearmanr(m["score_a"], m["score_b"]).statistic
        kendall = stats.kendalltau(m["score_a"], m["score_b"]).statistic

        # Weighted kappa on integer bins 0..10.
        xa = np.rint(m["score_a"].to_numpy()).astype(int)
        xb = np.rint(m["score_b"].to_numpy()).astype(int)
        xa = np.clip(xa, 0, 10)
        xb = np.clip(xb, 0, 10)

        table = pd.crosstab(xa, xb)
        # Ensure full 0..10 table.
        idx = list(range(11))
        table = table.reindex(index=idx, columns=idx, fill_value=0)
        kappa = cohens_kappa(table.to_numpy(), wt="quadratic").kappa

        rows.append(
            {
                "eval_a": a,
                "eval_b": b,
                "n_overlap": n,
                "spearman": float(spearman),
                "kendall": float(kendall),
                "weighted_kappa": float(kappa),
            }
        )

    return pd.DataFrame(rows)


def auto_human_calibration(
    df_scored: pd.DataFrame,
    key_df: pd.DataFrame,
    human_score_col: str = "score_overall",
    auto_score_col: str = "score_aggregated",
) -> dict[str, Any]:
    d = _normalize_scores(df_scored, score_col=human_score_col)
    if "blind_id" not in key_df.columns:
        return {"warning": "key file missing blind_id column"}
    if auto_score_col not in key_df.columns:
        return {"warning": f"key file missing {auto_score_col} column"}

    # Consolidate repeated human labels per blind_id into a robust central tendency.
    human = (
        d.groupby("blind_id", as_index=False)[human_score_col]
        .median()
        .rename(columns={human_score_col: "human_score"})
    )
    key = key_df[["blind_id", auto_score_col]].copy().rename(columns={auto_score_col: "auto_score"})
    key["auto_score"] = pd.to_numeric(key["auto_score"], errors="coerce")

    merged = human.merge(key, on="blind_id", how="inner").dropna(subset=["human_score", "auto_score"])
    n = len(merged)
    if n < 3:
        return {"warning": f"insufficient overlap for auto_vs_human correlation (n={n})"}

    spearman = stats.spearmanr(merged["human_score"], merged["auto_score"]).statistic
    kendall = stats.kendalltau(merged["human_score"], merged["auto_score"]).statistic
    pearson = stats.pearsonr(merged["human_score"], merged["auto_score"]).statistic
    mae = float(np.mean(np.abs(merged["human_score"] - merged["auto_score"])))

    return {
        "n_items": int(n),
        "human_aggregation": "median_by_blind_id",
        "spearman": float(spearman),
        "kendall": float(kendall),
        "pearson": float(pearson),
        "mae": mae,
    }


def agreement_markdown_report(
    scored_csv_path: str | Path,
    out_md_path: str | Path,
    key_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(scored_csv_path)
    pair_df = pairwise_agreement(df)
    calib: dict[str, Any] | None = None

    if key_csv_path is not None:
        key_df = pd.read_csv(key_csv_path)
        calib = auto_human_calibration(df, key_df)

    lines = []
    lines.append("# Human Evaluation Agreement Report")
    lines.append("")
    lines.append(f"Input rows: {len(df)}")
    lines.append("")
    lines.append("## Pairwise agreement")
    lines.append("")
    if len(pair_df) == 0:
        lines.append("No pairwise overlap found.")
    else:
        lines.append(pair_df.to_markdown(index=False))
        lines.append("")
        lines.append("Interpretation guide:")
        lines.append("- weighted_kappa >= 0.75: excellent")
        lines.append("- 0.60 to 0.74: good")
        lines.append("- 0.40 to 0.59: moderate")
        lines.append("- < 0.40: weak")

    if calib is not None:
        lines.append("")
        lines.append("## Auto vs Human Calibration")
        lines.append("")
        for k, v in calib.items():
            lines.append(f"- {k}: {v}")

    Path(out_md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md_path).write_text("\n".join(lines), encoding="utf-8")

    return pair_df
