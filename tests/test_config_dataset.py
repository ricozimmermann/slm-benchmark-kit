from __future__ import annotations

import textwrap

import pytest

from slm_benchmark.config import load_config
from slm_benchmark.dataset import load_jsonl_dataset


def test_load_config_success(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            random_seed: 123
            base_url: http://localhost:11434
            models: [m1]
            temperatures: [0.2]
            top_p: [0.9]
            top_k: [40]
            repetitions: 2
            max_tokens: 128
            timeout_seconds: 10
            dataset_path: datasets/input.jsonl
            eval_split: test
            output_path: results/out.jsonl
            judges:
              - type: heuristic
            trial_limit: 3
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_file)

    assert cfg.random_seed == 123
    assert cfg.models == ["m1"]
    assert cfg.eval_split == "test"
    assert cfg.trial_limit == 3


def test_load_config_invalid_trial_limit(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            models: [m1]
            temperatures: [0.2]
            top_p: [0.9]
            top_k: [40]
            dataset_path: datasets/input.jsonl
            output_path: results/out.jsonl
            trial_limit: 0
            """
        ).strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trial_limit"):
        load_config(cfg_file)


def test_load_jsonl_dataset_normalizes_split(tmp_path):
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        "\n".join(
            [
                '{"id":"1","split":"TEST","task_type":"bug_detection","difficulty":"easy","prompt":"p1","reference":"r1"}',
                '{"id":"2","task_type":"refactoring","prompt":"p2"}',
            ]
        ),
        encoding="utf-8",
    )

    rows = load_jsonl_dataset(data_file)

    assert len(rows) == 2
    assert rows[0].split == "test"
    assert rows[1].split == "test"
    assert rows[1].reference == ""
