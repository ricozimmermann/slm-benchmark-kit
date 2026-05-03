from __future__ import annotations

from itertools import combinations
import math
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
    required_columns = {"model", "score_aggregated"}
    if df.empty or not required_columns.issubset(df.columns):
        return pd.DataFrame()

    rows = []
    for model, g in df.groupby("model"):
        scored = pd.to_numeric(g["score_aggregated"], errors="coerce").dropna()
        vals = scored.to_numpy(dtype=float)
        if len(vals) > 0:
            ci_low, ci_high = bootstrap_ci_mean(vals)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            median_v = float(np.median(vals))
        else:
            ci_low, ci_high = float("nan"), float("nan")
            mean_v, std_v, median_v = float("nan"), float("nan"), float("nan")

        errors = g["error"].fillna("").astype(str) if "error" in g.columns else pd.Series([""] * len(g))
        error_rate = float((errors != "").mean())
        timeout_rate = float(errors.str.contains("timeout", case=False, regex=False).mean())
        if "judge_valid_count" in g.columns:
            judge_valid = pd.to_numeric(g["judge_valid_count"], errors="coerce")
        else:
            judge_valid = pd.Series(np.zeros(len(g), dtype=float), index=g.index)
        judge_all_failed_rate = float((judge_valid <= 0).mean())

        latency_p50 = float("nan")
        latency_p95 = float("nan")
        if "latency_ms" in g.columns:
            latency_vals = pd.to_numeric(g["latency_ms"], errors="coerce").dropna()
            if not latency_vals.empty:
                latency_p50 = float(np.percentile(latency_vals, 50))
                latency_p95 = float(np.percentile(latency_vals, 95))

        valid_rate = float("nan")
        if "valid_response" in g.columns:
            valid_rate = float(pd.to_numeric(g["valid_response"], errors="coerce").mean())

        rows.append(
            {
                "model": model,
                "n": len(g),
                "n_scored": int(len(scored)),
                "mean": mean_v,
                "std": std_v,
                "median": median_v,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "latency_p50_ms": latency_p50,
                "latency_p95_ms": latency_p95,
                "valid_rate": valid_rate,
                "error_rate": error_rate,
                "timeout_rate": timeout_rate,
                "judge_all_failed_rate": judge_all_failed_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_num = (na - 1) * var_a + (nb - 1) * var_b
    pooled_den = na + nb - 2
    if pooled_den <= 0:
        return float("nan")

    pooled_std = math.sqrt(pooled_num / pooled_den)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def welch_between_models(df: pd.DataFrame) -> dict:
    if df.empty or "model" not in df.columns or "score_aggregated" not in df.columns:
        return {"warning": "Welch test skipped because benchmark results are missing required columns."}

    models = list(df["model"].unique())
    if len(models) != 2:
        return {"warning": "Welch test requires exactly 2 models."}

    a = pd.to_numeric(df[df["model"] == models[0]]["score_aggregated"], errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(df[df["model"] == models[1]]["score_aggregated"], errors="coerce").dropna().to_numpy(dtype=float)

    if len(a) < 2 or len(b) < 2:
        return {"warning": "Welch test requires at least 2 scored samples per model."}

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    effect_size = cohens_d(a, b)
    return {
        "model_a": models[0],
        "model_b": models[1],
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "effect_size_cohens_d": effect_size,
        "significant_0_05": bool(p_value < 0.05),
    }


def fit_ols(df: pd.DataFrame):
    d = df.copy()
    required_columns = {"score_aggregated", "model", "task_type", "temperature", "top_p", "top_k"}
    if d.empty or not required_columns.issubset(d.columns):
        return None

    d["score_aggregated"] = pd.to_numeric(d["score_aggregated"], errors="coerce")
    d = d.dropna(subset=["score_aggregated"])
    if len(d) < 8:
        return None

    d["model"] = d["model"].astype("category")
    d["task_type"] = d["task_type"].astype("category")
    model = smf.ols(
        "score_aggregated ~ C(model) + temperature + top_p + top_k + C(task_type)",
        data=d,
    )
    if "item_id" in d.columns and d["item_id"].nunique() > 1:
        n_clusters = int(d["item_id"].nunique())
        n_constraints = len(model.exog_names) - 1
        if n_clusters <= n_constraints:
            return model.fit(cov_type="HC3")
        try:
            return model.fit(cov_type="cluster", cov_kwds={"groups": d["item_id"]})
        except Exception:
            return model.fit(cov_type="HC3")
    return model.fit(cov_type="HC3")


def ols_diagnostics(df: pd.DataFrame) -> dict:
    """Run Shapiro-Wilk, Breusch-Pagan and interaction OLS on existing data."""
    from statsmodels.stats.diagnostic import het_breuschpagan

    d = df.copy()
    required_columns = {"score_aggregated", "model", "task_type", "temperature", "top_p", "top_k"}
    if d.empty or not required_columns.issubset(d.columns):
        return {"warning": "Diagnostics skipped: missing required columns."}

    d["score_aggregated"] = pd.to_numeric(d["score_aggregated"], errors="coerce")
    d = d.dropna(subset=["score_aggregated"])
    if len(d) < 8:
        return {"warning": "Diagnostics skipped: insufficient data."}

    d["model"] = d["model"].astype("category")
    d["task_type"] = d["task_type"].astype("category")

    # --- base model (same as fit_ols) for residuals ---
    base_formula = "score_aggregated ~ C(model) + temperature + top_p + top_k + C(task_type)"
    base_fit = smf.ols(base_formula, data=d).fit()
    residuals = base_fit.resid.to_numpy()
    exog = base_fit.model.exog

    # Shapiro-Wilk (subsample to 5000 if needed; SW requires n <= 5000)
    sw_sample = residuals if len(residuals) <= 5000 else residuals[:5000]
    sw_stat, sw_p = stats.shapiro(sw_sample)

    # Breusch-Pagan
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, exog)

    # --- interaction model: task_type * temperature ---
    interaction_formula = (
        "score_aggregated ~ C(model) + C(task_type) * temperature + top_p + top_k"
    )
    interaction_fit = smf.ols(interaction_formula, data=d).fit(cov_type="HC3")

    return {
        "n_obs": int(len(d)),
        # Shapiro-Wilk
        "shapiro_wilk_stat": float(sw_stat),
        "shapiro_wilk_p": float(sw_p),
        "shapiro_wilk_note": (
            "Resíduos aproximadamente normais (p >= 0.05)"
            if sw_p >= 0.05
            else "Desvio de normalidade detectado (p < 0.05)"
        ),
        # Breusch-Pagan
        "breusch_pagan_lm": float(bp_lm),
        "breusch_pagan_lm_p": float(bp_lm_p),
        "breusch_pagan_f": float(bp_f),
        "breusch_pagan_f_p": float(bp_f_p),
        "breusch_pagan_note": (
            "Heterocedasticidade detectada (p < 0.05) — erros HC3 justificados"
            if bp_lm_p < 0.05
            else "Homocedasticidade não rejeitada (p >= 0.05)"
        ),
        # Interaction model
        "interaction_r2": float(interaction_fit.rsquared),
        "interaction_r2_adj": float(interaction_fit.rsquared_adj),
        "interaction_f_stat": float(interaction_fit.fvalue),
        "interaction_f_p": float(interaction_fit.f_pvalue),
        "interaction_summary": str(interaction_fit.summary()),
    }


def judge_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "judge_rationales" not in df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        rationales = row.get("judge_rationales", {}) or {}
        scores = row.get("judge_scores", {}) or {}
        if not isinstance(rationales, dict):
            continue
        for judge_name, rationale in rationales.items():
            status = "ok"
            rat = str(rationale or "")
            if rat.startswith("judge_parse_error"):
                status = "parse_error"
            elif rat.startswith("judge_error"):
                status = "judge_error"
            elif rat.startswith("judge_parse_fallback_used"):
                status = "parse_fallback"

            has_score = isinstance(scores, dict) and (judge_name in scores)
            rows.append(
                {
                    "judge": judge_name,
                    "has_score": int(has_score),
                    "parse_error": int(status == "parse_error"),
                    "judge_error": int(status == "judge_error"),
                    "parse_fallback": int(status == "parse_fallback"),
                }
            )

    if not rows:
        return pd.DataFrame()

    t = pd.DataFrame(rows)
    g = t.groupby("judge", as_index=False).agg(
        total_calls=("judge", "size"),
        valid_scores=("has_score", "sum"),
        parse_errors=("parse_error", "sum"),
        judge_errors=("judge_error", "sum"),
        parse_fallbacks=("parse_fallback", "sum"),
    )
    g["valid_rate"] = g["valid_scores"] / g["total_calls"]
    g["parse_error_rate"] = g["parse_errors"] / g["total_calls"]
    g["judge_error_rate"] = g["judge_errors"] / g["total_calls"]
    g["parse_fallback_rate"] = g["parse_fallbacks"] / g["total_calls"]
    return g.sort_values("judge").reset_index(drop=True)


def judge_pairwise_agreement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "judge_scores" not in df.columns:
        return pd.DataFrame()

    expanded = pd.json_normalize(df["judge_scores"])
    if expanded.empty or expanded.shape[1] < 2:
        return pd.DataFrame()

    expanded = expanded.apply(pd.to_numeric, errors="coerce")
    rows = []
    for a, b in combinations(expanded.columns.tolist(), 2):
        pair = expanded[[a, b]].dropna()
        n = len(pair)
        if n < 3:
            rows.append(
                {
                    "judge_a": a,
                    "judge_b": b,
                    "n_overlap": n,
                    "spearman": float("nan"),
                    "kendall": float("nan"),
                    "pearson": float("nan"),
                    "mae": float("nan"),
                }
            )
            continue

        a_constant = pair[a].nunique(dropna=True) <= 1
        b_constant = pair[b].nunique(dropna=True) <= 1
        if a_constant or b_constant:
            rows.append(
                {
                    "judge_a": a,
                    "judge_b": b,
                    "n_overlap": n,
                    "spearman": float("nan"),
                    "kendall": float("nan"),
                    "pearson": float("nan"),
                    "mae": float(np.mean(np.abs(pair[a] - pair[b]))),
                }
            )
            continue

        try:
            spearman = stats.spearmanr(pair[a], pair[b]).statistic
        except Exception:
            spearman = float("nan")
        try:
            kendall = stats.kendalltau(pair[a], pair[b]).statistic
        except Exception:
            kendall = float("nan")
        try:
            pearson = stats.pearsonr(pair[a], pair[b]).statistic
        except Exception:
            pearson = float("nan")

        mae = float(np.mean(np.abs(pair[a] - pair[b])))
        rows.append(
            {
                "judge_a": a,
                "judge_b": b,
                "n_overlap": n,
                "spearman": float(spearman),
                "kendall": float(kendall),
                "pearson": float(pearson),
                "mae": mae,
            }
        )

    return pd.DataFrame(rows)


def generate_markdown_report(df: pd.DataFrame, out_md: Path) -> None:
    summary = summarize_by_model(df)
    welch = welch_between_models(df)
    ols = fit_ols(df)
    diagnostics = ols_diagnostics(df)
    judge_agreement = judge_pairwise_agreement(df)
    judge_health = judge_health_summary(df)

    lines = []
    lines.append("# SLM Benchmark Report")
    lines.append("")
    lines.append(f"Total trials: {len(df)}")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    if summary.empty:
        lines.append("No benchmark rows available to summarize.")
    else:
        lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Welch t-test")
    lines.append("")
    for k, v in welch.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## OLS (cluster-robust by item when available)")
    lines.append("")
    if ols is None:
        lines.append("Skipped OLS: not enough scored samples after filtering invalid judge outputs.")
    else:
        lines.append("```text")
        lines.append(str(ols.summary()))
        lines.append("```")

    lines.append("")
    lines.append("## OLS Diagnostics")
    lines.append("")
    if "warning" in diagnostics:
        lines.append(diagnostics["warning"])
    else:
        lines.append("### Shapiro-Wilk (normalidade dos resíduos)")
        lines.append("")
        lines.append(f"- stat: {diagnostics['shapiro_wilk_stat']:.6f}")
        lines.append(f"- p-value: {diagnostics['shapiro_wilk_p']:.6e}")
        lines.append(f"- interpretação: {diagnostics['shapiro_wilk_note']}")
        lines.append("")
        lines.append("### Breusch-Pagan (homocedasticidade)")
        lines.append("")
        lines.append(f"- LM stat: {diagnostics['breusch_pagan_lm']:.6f}")
        lines.append(f"- LM p-value: {diagnostics['breusch_pagan_lm_p']:.6e}")
        lines.append(f"- F stat: {diagnostics['breusch_pagan_f']:.6f}")
        lines.append(f"- F p-value: {diagnostics['breusch_pagan_f_p']:.6e}")
        lines.append(f"- interpretação: {diagnostics['breusch_pagan_note']}")
        lines.append("")
        lines.append("### OLS com interação task_type × temperature")
        lines.append("")
        lines.append(f"- R²: {diagnostics['interaction_r2']:.4f}")
        lines.append(f"- R² ajustado: {diagnostics['interaction_r2_adj']:.4f}")
        lines.append(f"- F-stat: {diagnostics['interaction_f_stat']:.4f}")
        lines.append(f"- F p-value: {diagnostics['interaction_f_p']:.6e}")
        lines.append("")
        lines.append("```text")
        lines.append(diagnostics["interaction_summary"])
        lines.append("```")

    lines.append("")
    lines.append("## Judge Agreement (Pairwise)")
    lines.append("")
    if judge_agreement.empty:
        lines.append("Not enough judge score columns to compute pairwise agreement.")
    else:
        lines.append(judge_agreement.to_markdown(index=False))

    lines.append("")
    lines.append("## Judge Health")
    lines.append("")
    if judge_health.empty:
        lines.append("No judge rationale data available to compute health metrics.")
    else:
        lines.append(judge_health.to_markdown(index=False))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
