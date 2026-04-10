from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(sample.mean())
    low = np.percentile(means, (1 - ci) / 2 * 100)
    high = np.percentile(means, (1 + ci) / 2 * 100)
    return float(low), float(high)


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        vals = g["score_aggregated"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci_mean(vals)
        rows.append(
            {
                "model": model,
                "n": len(g),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "latency_p50_ms": float(np.percentile(g["latency_ms"], 50)),
                "latency_p95_ms": float(np.percentile(g["latency_ms"], 95)),
                "valid_rate": float(g["valid_response"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def welch_between_models(df: pd.DataFrame) -> dict:
    models = list(df["model"].unique())
    if len(models) != 2:
        return {"warning": "Welch test requires exactly 2 models."}

    a = df[df["model"] == models[0]]["score_aggregated"].to_numpy(dtype=float)
    b = df[df["model"] == models[1]]["score_aggregated"].to_numpy(dtype=float)

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return {
        "model_a": models[0],
        "model_b": models[1],
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_0_05": bool(p_value < 0.05),
    }


def fit_ols(df: pd.DataFrame):
    d = df.copy()
    d["model"] = d["model"].astype("category")
    d["task_type"] = d["task_type"].astype("category")
    return smf.ols(
        "score_aggregated ~ C(model) + temperature + top_p + top_k + C(task_type)",
        data=d,
    ).fit()


def generate_markdown_report(df: pd.DataFrame, out_md: Path) -> None:
    summary = summarize_by_model(df)
    welch = welch_between_models(df)
    ols = fit_ols(df)

    lines = []
    lines.append("# SLM Benchmark Report")
    lines.append("")
    lines.append(f"Total trials: {len(df)}")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Welch t-test")
    lines.append("")
    for k, v in welch.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## OLS (real coefficients)")
    lines.append("")
    lines.append("```text")
    lines.append(str(ols.summary()))
    lines.append("```")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
