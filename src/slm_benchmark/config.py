from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkConfig:
    random_seed: int
    base_url: str
    models: list[str]
    temperatures: list[float]
    top_p: list[float]
    top_k: list[int]
    repetitions: int
    max_tokens: int
    timeout_seconds: int
    dataset_path: Path
    eval_split: str
    output_path: Path
    judges: list[dict[str, Any]]
    trial_limit: int | None


def load_config(path: str | Path) -> BenchmarkConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must define a mapping/object at the top level")

    trial_limit_raw = raw.get("trial_limit")
    trial_limit = int(trial_limit_raw) if trial_limit_raw is not None else None
    if trial_limit is not None and trial_limit <= 0:
        raise ValueError("trial_limit must be > 0 when provided")

    eval_split = str(raw.get("eval_split", "test")).strip().lower()
    if eval_split not in {"train", "dev", "test", "all"}:
        raise ValueError("eval_split must be one of: train, dev, test, all")

    return BenchmarkConfig(
        random_seed=int(raw.get("random_seed", 42)),
        base_url=str(raw.get("base_url", "http://localhost:11434")),
        models=list(raw["models"]),
        temperatures=[float(x) for x in raw["temperatures"]],
        top_p=[float(x) for x in raw["top_p"]],
        top_k=[int(x) for x in raw["top_k"]],
        repetitions=int(raw.get("repetitions", 3)),
        max_tokens=int(raw.get("max_tokens", 256)),
        timeout_seconds=int(raw.get("timeout_seconds", 90)),
        dataset_path=Path(raw["dataset_path"]),
        eval_split=eval_split,
        output_path=Path(raw["output_path"]),
        judges=list(raw.get("judges", [{"type": "heuristic"}])),
        trial_limit=trial_limit,
    )
