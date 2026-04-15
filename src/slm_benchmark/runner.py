from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from .clients import OllamaClient
from .config import BenchmarkConfig
from .dataset import load_jsonl_dataset
from .judges import build_judges


@dataclass
class Trial:
    model: str
    item_id: str
    split: str
    task_type: str
    difficulty: str
    temperature: float
    top_p: float
    top_k: int
    repetition: int
    prompt: str
    reference: str


@dataclass
class TrialResult:
    timestamp_utc: str
    model: str
    item_id: str
    split: str
    task_type: str
    difficulty: str
    temperature: float
    top_p: float
    top_k: int
    repetition: int
    latency_ms: int
    response_text: str
    response_chars: int
    valid_response: bool
    error: str | None
    judge_scores: dict[str, float]
    judge_rationales: dict[str, str]
    judge_valid_count: int
    judge_invalid_count: int
    score_aggregated: float | None


def _make_trials(cfg: BenchmarkConfig) -> list[Trial]:
    items = load_jsonl_dataset(cfg.dataset_path)
    if cfg.eval_split != "all":
        items = [item for item in items if item.split == cfg.eval_split]
    if not items:
        raise ValueError(f"No dataset items found for eval_split='{cfg.eval_split}'")

    trials: list[Trial] = []

    for model in cfg.models:
        for item in items:
            for temperature in cfg.temperatures:
                for top_p in cfg.top_p:
                    for top_k in cfg.top_k:
                        for repetition in range(1, cfg.repetitions + 1):
                            trials.append(
                                Trial(
                                    model=model,
                                    item_id=item.id,
                                    split=item.split,
                                    task_type=item.task_type,
                                    difficulty=item.difficulty,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    repetition=repetition,
                                    prompt=item.prompt,
                                    reference=item.reference,
                                )
                            )
    return trials


def run_benchmark(cfg: BenchmarkConfig) -> Path:
    random.seed(cfg.random_seed)

    client = OllamaClient(base_url=cfg.base_url, timeout_seconds=cfg.timeout_seconds)
    judges = build_judges(cfg.judges, client)

    trials = _make_trials(cfg)
    random.shuffle(trials)
    if cfg.trial_limit is not None:
        trials = trials[: cfg.trial_limit]

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    with cfg.output_path.open("w", encoding="utf-8") as out_f:
        for idx, trial in enumerate(trials, start=1):
            out = client.generate(
                model=trial.model,
                prompt=trial.prompt,
                temperature=trial.temperature,
                top_p=trial.top_p,
                top_k=trial.top_k,
                max_tokens=cfg.max_tokens,
            )

            judge_scores: dict[str, float] = {}
            judge_rationales: dict[str, str] = {}
            judge_valid_count = 0
            judge_invalid_count = 0

            if out.error is None and out.text.strip():
                for judge in judges:
                    j = judge.score(out.text, trial.reference)
                    judge_rationales[j.name] = j.rationale
                    if j.score is None:
                        judge_invalid_count += 1
                        continue
                    judge_scores[j.name] = float(j.score)
                    judge_valid_count += 1

                agg = float(median(judge_scores.values())) if judge_scores else None
                valid_response = judge_valid_count > 0
            else:
                agg = None
                valid_response = False
                judge_invalid_count = len(judges)

            result = TrialResult(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                model=trial.model,
                item_id=trial.item_id,
                split=trial.split,
                task_type=trial.task_type,
                difficulty=trial.difficulty,
                temperature=trial.temperature,
                top_p=trial.top_p,
                top_k=trial.top_k,
                repetition=trial.repetition,
                latency_ms=out.latency_ms,
                response_text=out.text,
                response_chars=len(out.text),
                valid_response=valid_response,
                error=out.error,
                judge_scores=judge_scores,
                judge_rationales=judge_rationales,
                judge_valid_count=judge_valid_count,
                judge_invalid_count=judge_invalid_count,
                score_aggregated=agg,
            )

            out_f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            score_label = f"{agg:.2f}" if agg is not None else "NA"
            print(
                f"[{idx}/{len(trials)}] {trial.model} {trial.item_id} "
                f"score={score_label} latency={out.latency_ms}ms"
            )

    return cfg.output_path
