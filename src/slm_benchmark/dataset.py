from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetItem:
    id: str
    split: str
    task_type: str
    difficulty: str
    prompt: str
    reference: str


def load_jsonl_dataset(path: str | Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            items.append(
                DatasetItem(
                    id=str(raw["id"]),
                    split=str(raw.get("split", "test")).strip().lower(),
                    task_type=str(raw["task_type"]),
                    difficulty=str(raw.get("difficulty", "unknown")),
                    prompt=str(raw["prompt"]),
                    reference=str(raw.get("reference", "")),
                )
            )
    return items
