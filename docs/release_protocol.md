# Scientific Release Protocol

This protocol ensures each benchmark release is reproducible and auditable.

## 1) Versioning rules

- Benchmark versions follow SemVer: MAJOR.MINOR.PATCH.
- MAJOR: breaking protocol changes (dataset schema, scoring logic, trial design).
- MINOR: non-breaking methodological additions (new tasks, new metrics).
- PATCH: bug fixes that do not change scientific conclusions.

Files to update every release:
- VERSION
- CHANGELOG.md
- configs/benchmark_ollama.yaml (if config changed)
- datasets/slm_tasks_ptbr.jsonl (if dataset changed)

## 2) Required release artifacts

For each release tag, publish:
- Raw benchmark output JSONL.
- Summary report markdown.
- Human eval assignment template used.
- Human agreement report.
- Exact runtime config YAML.
- Git commit hash and platform metadata.

## 3) Methodology guardrails

- Keep holdout evaluation split untouched during tuning.
- Use fixed seed for each release.
- Record model IDs exactly as run in Ollama.
- Keep blind mapping file private during human scoring.
- Do not alter scored human files after agreement analysis.

## 4) Release checklist

- [ ] Bump VERSION.
- [ ] Append CHANGELOG with method/data changes.
- [ ] Run benchmark with release config.
- [ ] Generate statistical report.
- [ ] Prepare blind human evaluation sample.
- [ ] Collect human scores and run agreement report.
- [ ] Archive artifacts in results/release-vX.Y.Z/.
- [ ] Tag release in git: vX.Y.Z.

## 5) Suggested release folder

results/release-vX.Y.Z/
- raw_benchmark.jsonl
- report.md
- config.yaml
- human_assignment.csv
- human_key_private.csv
- human_scored.csv
- human_agreement.md
- metadata.json
