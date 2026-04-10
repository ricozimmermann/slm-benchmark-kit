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
    output_path: Path
    judges: list[dict[str, Any]]


def load_config(path: str | Path) -> BenchmarkConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

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
        output_path=Path(raw["output_path"]),
        judges=list(raw.get("judges", [{"type": "heuristic"}])),
    )
