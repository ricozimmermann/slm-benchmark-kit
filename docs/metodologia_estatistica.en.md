# Statistical Methodology for the SLM Benchmark Kit

## 0. Scientific rationale: why SLMs and why benchmarking

Interest in Small Language Models (SLMs) is aligned with a shift in applied AI: moving away from approaches based exclusively on maximum scale and toward efficient models that can run under real constraints on cost, energy, latency, and privacy.

In practice, SLMs are especially relevant because they:
- reduce memory and compute requirements, enabling use on edge/mobile devices and in limited-infrastructure environments;
- allow local/offline processing, with less dependence on the cloud and lower exposure of sensitive data;
- support technological democratization by widening access for institutions with less compute capacity;
- can deliver competitive performance in specific domains when properly tuned and evaluated.

However, greater efficiency does not automatically imply better utility. For that reason, model comparison must be done under a controlled experimental protocol, with explicit statistical inference and complementary operational metrics.

The rationale for benchmarking in this project is therefore twofold:
1. methodological: measure performance differences while controlling uncertainty and confounding;
2. applied: identify the best trade-off among quality, stability, latency, and compute cost for real-world scenarios.

In short, SLMs are strategic because of efficiency and accessibility; reproducible benchmarking is strategic because it makes technical decisions reliable.

## 1. Executive summary

This document formalizes the statistical content used in the `slm-benchmark-kit` project to compare SLMs with a focus on:
- estimation with explicit uncertainty;
- inferential comparison between models;
- control of experimental covariates;
- quality diagnosis of the automatic evaluation system (judges).

The current pipeline combines:
- descriptive statistics per model;
- bootstrap confidence intervals for the mean score;
- Welch's t-test for pairwise model comparison;
- Cohen's d effect size;
- OLS regression with robust errors (cluster by item when available; HC3 as fallback);
- agreement and health metrics for judges.

## 2. Analysis unit and experimental design

## 2.1 Observational unit

The observational unit is the `trial` recorded in the results JSONL file, containing:
- model identifier (`model`);
- item identifier (`item_id`), task type, and difficulty;
- generation hyperparameters (`temperature`, `top_p`, `top_k`);
- repetition (`repetition`);
- aggregated score (`score_aggregated`) when at least one judge is valid.

## 2.2 Operational definitions

- `valid_response`: in the current pipeline, this means that at least one judge returned a valid score (`judge_valid_count > 0`).
- `score_aggregated`: median of the valid judge scores for the trial.
- `error`: generation error (for example, timeout). When present, the trial usually has no aggregated score.

## 2.3 Strategies to reduce bias

The benchmark design includes:
- fixed seed for reproducibility;
- randomized execution order of trials;
- repeated runs per parameter combination;
- separation by evaluation split (`eval_split`) to avoid leakage between tuning and testing;
- evaluation by multiple judges with robust median aggregation.

## 3. Reported variables and metrics

## 3.1 Main performance

For each model, the report includes:
- `n`: total number of trials;
- `n_scored`: number of trials with numeric `score_aggregated`;
- mean, sample standard deviation, and median of `score_aggregated`;
- 95% bootstrap confidence interval for the mean (`ci95_low`, `ci95_high`).

## 3.2 Operational reliability

The report also includes:
- `valid_rate` (mean of `valid_response`);
- `error_rate` (proportion of non-empty errors);
- `timeout_rate` (proportion of errors containing "timeout");
- `judge_all_failed_rate` (proportion of trials with `judge_valid_count <= 0`);
- `latency_p50_ms` and `latency_p95_ms`.

These metrics prevent conclusions based only on average score and make cost/stability visible.

## 4. Statistical inference methods

## 4.1 Bootstrap confidence interval for the mean

For each model, the 95% CI for the mean is estimated with percentile bootstrap:
1. sample with replacement, `n_boot = 5000`, from the vector of valid scores;
2. compute the mean on each resample;
3. use the 2.5th and 97.5th percentiles of the bootstrap distribution.

Current configuration:
- `seed = 42` for bootstrap.

Interpretation:
- a narrower CI indicates a more precise mean estimate;
- overlapping CIs do not replace a formal hypothesis test.

## 4.2 Welch's t-test (pairwise model comparison)

When there are exactly two models with at least 2 valid observations per group, Welch's t-test is applied (unequal variances):

- Null hypothesis: population means are equal.
- Alternative hypothesis: population means are different.

The report includes:
- `t_stat`;
- `p_value`;
- `significant_0_05` indicator.

Scope note:
- the current implementation runs Welch only when there are 2 models; for more than 2 models, the test is skipped with a warning.

## 4.3 Effect size (Cohen's d)

Alongside Welch, Cohen's d is computed to quantify effect magnitude:

- d ~ 0.2: small effect (rule of thumb);
- d ~ 0.5: medium effect;
- d ~ 0.8: large effect.

Good practice:
- interpret `p_value` together with `d`;
- also report the mean difference (`mean_a - mean_b`) and the application context.

## 4.4 OLS regression with robust errors

To control for configuration confounding and task composition, the project fits:

`score_aggregated ~ C(model) + temperature + top_p + top_k + C(task_type)`

Details:
- `model` and `task_type` enter as categorical factors;
- the fit is performed only if the required columns are present and there are at least 8 valid observations;
- if `item_id` has more than one unique value, cluster-robust covariance by item is used;
- otherwise (or if fitting fails), HC3 is used.

Recommended interpretation:
- `C(model)` coefficients represent conditional differences (adjusted for covariates);
- robust errors reduce sensitivity to heteroskedasticity;
- clustering by item helps handle within-item dependence across repetitions.

## 5. Quality of the automatic evaluation system (judges)

## 5.1 Pairwise agreement

From `judge_scores`, the system computes for each judge pair:
- Spearman;
- Kendall tau;
- Pearson;
- MAE (mean absolute error);
- `n_overlap` (samples with valid scores in both judges).

Edge-case handling:
- if `n_overlap < 3`, correlations are reported as `NaN`;
- if a series is constant, correlations are `NaN` and MAE is still computed.

## 5.2 Judge health

From `judge_rationales`, the system estimates per judge:
- `valid_rate`;
- `parse_error_rate`;
- `judge_error_rate`;
- `parse_fallback_rate`.

These metrics separate:
- criterion disagreement (low agreement);
- technical parse/infrastructure failure (poor operational health).

## 6. Threats to validity and limitations

## 6.1 Internal validity

- Dependence between trials: repeated runs of the same `item_id` are not strictly independent (partly mitigated by cluster-robust OLS).
- Selection on valid responses: score analyses use non-null `score_aggregated`; different failure rates across models can bias mean comparisons.

## 6.2 External validity

- Results depend on the current dataset (`slm_tasks_ptbr.jsonl`), language, and task distribution.
- Generalization to other domains requires replication with new stratified datasets.

## 6.3 Construct validity

- The aggregated score depends on the judge rubric (heuristic/SLM or LLM) and the robustness of parsing.
- High agreement does not imply full semantic validity; blind human evaluation is recommended as a complement.

## 7. Reproducibility and scientific reporting

For auditable reporting, always publish:
- raw JSONL;
- markdown report generated by the pipeline;
- exact YAML configuration;
- commit hash;
- dataset version and checksum;
- protocol version (`VERSION` + `CHANGELOG.md`).

Additional operational checklist:
1. run the benchmark with `eval_split: test` for final conclusions;
2. record failures (`error_rate`, `timeout_rate`) alongside the scores;
3. report effect + uncertainty (mean difference, CI, `p_value`, `d`);
4. avoid conclusions based solely on the 0.05 threshold.

## 8. Interpretation guidelines

When comparing models, prioritize this order:
1. operational feasibility (valid_rate, error_rate, timeout_rate, latency);
2. central estimate and uncertainty (mean + bootstrap CI);
3. inferential evidence (Welch + effect size);
4. adjusted analysis (robust OLS) for conclusion robustness.

A model with a higher mean but a high failure rate can be worse in practice than a model with a slightly lower mean and better stability.

## 9. Methodological references (ABNT)

- CAMERON, A. Colin; MILLER, Douglas L. A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources, v. 50, n. 2, p. 317-372, 2015. DOI: https://doi.org/10.3368/jhr.50.2.317.
- COHEN, Jacob. Statistical Power Analysis for the Behavioral Sciences. 2. ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988. ISBN: 9780805802832.
- EFRON, Bradley; TIBSHIRANI, Robert J. An Introduction to the Bootstrap. New York: Chapman & Hall, 1993. ISBN: 9780412042317.
- MACKINNON, James G.; WHITE, Halbert. Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties. Journal of Econometrics, v. 29, n. 3, p. 305-325, 1985. DOI: https://doi.org/10.1016/0304-4076(85)90158-7.
- WELCH, B. L. The Generalization of Student's Problem when Several Different Population Variances are Involved. Biometrika, v. 34, n. 1-2, p. 28-35, 1947. DOI: https://doi.org/10.2307/2332510.

### 9.1 Traceability note

The references in this section were verified using bibliographic metadata from external sources:
- articles: Crossref (title, authors, journal, volume, issue, pages, and DOI);
- books: ISBN catalogs (Open Library/Google Books) for authors, edition, publisher, and year.

Verification date: 2026-04-15.

### 9.2 Evidence of use in Brazilian repositories

To address a possible question about the current relevance of the methods, an additional automated query was run against Brazilian institutional repositories. The goal was not to replace the foundational references, but to check recent use in theses and dissertations.

Observed summary:
- UFMG ([repositorio.ufmg.br](https://repositorio.ufmg.br/)): bootstrap search returned 580 results; Welch search returned 394 results.
- UFPE ([repositorio.ufpe.br](https://repositorio.ufpe.br/)): bootstrap search returned 1196 results; Welch search returned 766 results.
- UFF ([app.uff.br/riuff](https://app.uff.br/riuff/)): bootstrap search returned 34 results; Welch search returned 35 results.

Examples returned by the queries:
- UFMG: "GLARMA Model for Temporal Data Analysis: ... a bootstrap proposal for inference on model parameters" (2024).
- UFMG: "Caracterizacao de um modelo de sinucleinopatia ..." (2024), with explicit citation of Welch t test in the extracted abstract.
- UFPE: "Aplicacao de Metodos Bootstrap na Construcao de Intervalos de Confianca para os parametros da Distribuicao Gama" (2022).
- UFF: "Concentracoes sericas de 25-hidroxivitamina D ..." (2025), with use of Student's t-test or Welch in the extracted abstract.

Interpretation:
- classical methods (bootstrap, Welch, effect size, and robust errors) remain widely used in recent research;
- therefore, keeping foundational references is methodologically appropriate;
- adding recent usage evidence improves the justification for current relevance.

Collection limitations:
- USP, UNICAMP, and UFRGS did not yield stable structured extraction through the automatic tool used;
- UNIFESP returned a new DSpace structure and would require refined search routes for term-based extraction;
- FGV redirected to a different institutional portal, also requiring assisted manual search;
- UFABC points to a general bibliographic catalog rather than a thesis repository in the same DSpace pattern, which requires a different query strategy.

---

Scope of this version:
- This document describes the statistical methodology implemented in the current code at `src/slm_benchmark/analysis.py` and its integration with result generation in `scripts/analyze_results.py`.