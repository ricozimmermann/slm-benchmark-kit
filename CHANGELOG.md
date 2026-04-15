# Changelog

All notable methodology and tooling changes are tracked in this file.

## [0.2.0] - 2026-04-14

### Added
- Explicit evaluation split control via `eval_split` in benchmark config.
- Dataset item support for `split` field with backward-compatible default (`test`).
- Materialized train/dev/test split in `datasets/slm_tasks_ptbr.jsonl` (24/8/8).
- Operational robustness metrics in model summary (`error_rate`, `timeout_rate`).
- Welch output now includes effect size (`effect_size_cohens_d`).
- Optional auto_vs_human calibration section in agreement report using decode key (`--key`).

### Changed
- Runner now filters trials by configured evaluation split.
- OLS reporting now uses cluster-robust covariance by `item_id` when available (fallback HC3).
- Report section title updated to reflect robust OLS behavior.
- Scientific release protocol updated with dataset checksum and split-policy requirements.

## [0.1.0] - 2026-04-09

### Added
- Initial reusable benchmark framework for SLMs.
- Reproducible runner with fixed seed and randomized trial order.
- Multi-judge setup with aggregated score by median.
- Statistical analysis with Welch t-test, bootstrap CI, and OLS.
- Stratified dataset with 40 tasks across 5 task families.
- Blind human evaluation pipeline and agreement reporting.
- Scientific release protocol and report/paper templates.
