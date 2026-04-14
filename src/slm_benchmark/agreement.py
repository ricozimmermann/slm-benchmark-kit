from __future__ import annotations

from itertools import combinations
from pathlib import Path

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


def agreement_markdown_report(scored_csv_path: str | Path, out_md_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(scored_csv_path)
    pair_df = pairwise_agreement(df)

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

    Path(out_md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md_path).write_text("\n".join(lines), encoding="utf-8")

    return pair_df
