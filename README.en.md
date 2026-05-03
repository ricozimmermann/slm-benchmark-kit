# slm-benchmark-kit

Reusable benchmark framework for SLMs (Small Language Models).

Goal:
- compare models and parameters with reproducibility;
- support multiple projects with the same protocol;
- produce real statistical analysis (Welch t-test with effect size, item-robust OLS, bootstrap CI).

## 1. Methodological strategy

This project already includes essential benchmark improvements:
- fixed random seed for reproducibility;
- randomized trial order;
- repetitions per parameter combination;
- explicit evaluation split (`eval_split`) to separate tuning and testing;
- multi-judge setup with median aggregation, using two named LLM judges in the main benchmark;
- statistical analysis with scipy/statsmodels (no heuristic p-value shortcuts).

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -e .
```

## 2.1 Quick onboarding (recommended order)

If this is your first time in this project, follow this order:

1. Run a local smoke test to validate environment and Ollama:

```bash
python scripts/run_benchmark.py --config configs/benchmark_local_smoke.yaml --check-local
```

2. Run the full benchmark (or your custom config):

```bash
python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml
```

3. Generate the statistical report:

```bash
python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md
```

4. (Optional) Prepare blind human evaluation:

```bash
python scripts/prepare_human_eval.py \
	--input results/raw_benchmark.jsonl \
	--assignment results/human_assignment.csv \
	--key results/human_key_private.csv \
	--evaluators eval01 eval02 eval03 \
	--sample-size 120 \
	--overlap-rate 0.25
```

5. (Optional) Generate agreement among evaluators:

```bash
python scripts/agreement_report.py \
	--input results/human_assignment_scored.csv \
	--key results/human_key_private.csv \
	--output results/human_agreement.md
```

## 3. Run benchmark

Main command (already detailed in onboarding 2.1):
- `python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml`

### 3.1 Fast local test (SLM in Ollama)

Use the smoke configuration to validate the local pipeline with a small sample and compare two models at reduced scale:

```bash
python scripts/run_benchmark.py --config configs/benchmark_local_smoke.yaml --check-local
```

This does:
- local Ollama connectivity check;
- warning if configured model is not downloaded locally;
- execution with small `trial_limit` for fast validation.

Raw output (JSONL):
- `results/raw_benchmark_local_smoke.jsonl`

## 4. Generate report

Main command (already detailed in onboarding 2.1):
- `python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md`

## 5. Project architecture (didactic)

### 5.1 Visual flow map

```text
datasets/*.jsonl
	 -> configs/*.yaml
	 -> scripts/run_benchmark.py
	 -> src/slm_benchmark/runner.py
	 -> results/raw_benchmark*.jsonl
	 -> scripts/analyze_results.py
	 -> src/slm_benchmark/analysis.py
	 -> results/report.md

Optional (human evaluation):
results/raw_benchmark*.jsonl
	 -> scripts/prepare_human_eval.py
	 -> results/human_assignment.csv + results/human_key_private.csv
	 -> (evaluators fill scores)
	 -> scripts/agreement_report.py
	 -> results/human_agreement.md
```

### 5.2 Folder by folder

- `configs/`
	- Role: define the experiment without changing code.
	- `benchmark_local_smoke.yaml`: fast validation (few combinations, low `trial_limit`).
	- `benchmark_ollama.yaml`: main benchmark (more models/parameters and multi-judge).
	- Main fields: `models`, `temperatures`, `top_p`, `top_k`, `repetitions`, `dataset_path`, `eval_split`, `output_path`, `judges`.

- `datasets/`
	- Role: task source in JSONL (one task per line).
	- `slm_tasks_ptbr.jsonl`: initial dataset stratified into 5 families.
	- Practical item schema: `id`, `split`, `task_type`, `difficulty`, `prompt`, `reference`.

- `scripts/`
	- Role: CLI interface to run the end-to-end pipeline.
	- `run_benchmark.py`: loads config, optionally checks local Ollama, runs trials, saves raw JSONL.
	- `analyze_results.py`: runs statistical analysis and generates markdown report.
	- `prepare_human_eval.py`: builds blind package for human annotation with controlled overlap.
	- `agreement_report.py`: computes evaluator agreement (correlations + weighted kappa).

- `src/slm_benchmark/`
	- Role: benchmark core logic.
	- `config.py`: YAML parsing/validation into `BenchmarkConfig`.
	- `dataset.py`: robust JSONL dataset loading into `DatasetItem` objects.
	- `clients.py`: Ollama client (healthcheck, model listing, response generation).
	- `judges.py`: automatic evaluators (heuristic and LLM judge via Ollama).
	- `runner.py`: generates trial combinations, randomizes order, executes, aggregates median score, persists results.
	- `analysis.py`: model summary with operational metrics, bootstrap CI, Welch t-test with Cohen's d, item-robust OLS.
	- `human_eval.py`: stratified sampling and blind task assignment for humans.
	- `agreement.py`: inter-rater agreement report.

- `results/`
	- Role: generated execution artifacts.
	- Typical: `raw_benchmark*.jsonl`, `report.md`, `human_assignment.csv`, `human_agreement.md`.
	- Best practice: create subfolders per release (`results/release-vX.Y.Z/`).

- `docs/`
	- Role: methodological governance.
	- `human_eval_rubric.en.md`: human scoring criteria (0 to 10 by dimension).
	- `metodologia_estatistica.en.md`: formal definition of the statistical methodology (bootstrap CI, Welch, Cohen's d, robust OLS, reliability metrics).
	- `release_protocol.en.md`: scientific checklist for reproducible releases.

- `templates/`
	- Role: communication/scientific writing standards.
	- `report_template.md`: technical report template.
	- `paper_outline.md`: base paper structure.

- Root files
	- `README.md`: project operational guide.
	- `pyproject.toml`: Python packaging + dependencies.
	- `VERSION` and `CHANGELOG.md`: version control and history.
	- `LICENSE`: licensing.

### 5.3 How modules connect internally

1. `scripts/run_benchmark.py` calls `load_config` (`config.py`).
2. `runner.py` calls `load_jsonl_dataset` (`dataset.py`) and creates trials.
3. Each trial uses `OllamaClient.generate` (`clients.py`).
4. The response is evaluated by `build_judges` (`judges.py`) and aggregated by median.
5. The result is written as JSONL in `results/`.
6. `scripts/analyze_results.py` calls `analysis.py` and generates markdown statistics.
7. If human evaluation is used, `human_eval.py` and `agreement.py` close the loop.

## 6. Contribution and new features workflow

To evolve a repository that already exists, use a feature branch + Pull Request workflow:

```bash
git checkout -b feat/feature-name
# implement your changes and run the required tests
git add .
git commit -m "feat: short feature description"
git push -u origin feat/feature-name
```

Then open a Pull Request on GitHub to review and merge into `main`.

Recommended checklist before opening the PR:
1. run the local smoke test (`configs/benchmark_local_smoke.yaml`);
2. update `CHANGELOG.md` when the change is relevant;
3. validate whether `VERSION` should be bumped for release.

## 7. Stratified dataset

This project already includes an initial dataset with 40 tasks stratified into 5 families:
- code_explanation
- bug_detection
- refactoring
- test_generation
- security_performance

For academic protocol 0.2.0, the dataset has a materialized split in each item:
- `train`: 24 tasks
- `dev`: 8 tasks
- `test`: 8 tasks

File:
- datasets/slm_tasks_ptbr.jsonl

## 8. Blind human evaluation

Preparation and agreement commands are already listed in onboarding 2.1 (steps 4 and 5).

Rubric:
- docs/human_eval_rubric.en.md

### 8.1 Input validations (current)

The human evaluation flow now fails early with a clear message when:
- `--evaluators` is empty;
- `--sample-size` is less than or equal to zero;
- `--overlap-rate` is outside `[0, 1]`;
- input JSONL has no rows with `valid_response = true`.

## 9. Scientific release protocol

Versioning and methodological governance:
- VERSION
- CHANGELOG.md
- docs/release_protocol.en.md

Templates for dissemination:
- templates/report_template.md
- templates/paper_outline.md

## 10. Authorship and AI assistance

The idea, project architecture, and benchmark methodology were conceived and developed by the author.
The source code was written with AI assistance (GitHub Copilot / Claude), under continuous supervision and review by the author.

All AI-generated content was reviewed, validated, and adapted before being incorporated into the project.
