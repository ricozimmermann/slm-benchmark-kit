from __future__ import annotations

import json

from slm_benchmark.clients import GenerationOutput, OllamaClient
from slm_benchmark.config import BenchmarkConfig
from slm_benchmark.judges import OllamaJudge
from slm_benchmark.runner import _make_trials, run_benchmark


class _FakeJudge:
    name = "fake_judge"

    def score(self, response_text: str, reference: str):
        class _S:
            name = "fake_judge"
            score = 8.0
            rationale = "ok"

        return _S()


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, **kwargs):
        return GenerationOutput(text="Resposta de teste\ncom duas linhas\n.", latency_ms=12)


class _FakeNoScoreJudge:
    name = "no_score_judge"

    def score(self, response_text: str, reference: str):
        class _S:
            name = "no_score_judge"
            score = None
            rationale = "judge_parse_error: malformed"

        return _S()


def _write_dataset(path):
    path.write_text(
        "\n".join(
            [
                '{"id":"a1","split":"test","task_type":"bug_detection","difficulty":"easy","prompt":"p1","reference":"r1"}',
                '{"id":"a2","split":"dev","task_type":"refactoring","difficulty":"hard","prompt":"p2","reference":"r2"}',
            ]
        ),
        encoding="utf-8",
    )


def test_make_trials_applies_split_filter(tmp_path):
    ds = tmp_path / "dataset.jsonl"
    _write_dataset(ds)

    cfg = BenchmarkConfig(
        random_seed=42,
        base_url="http://localhost:11434",
        models=["m1"],
        temperatures=[0.0, 0.2],
        top_p=[0.9],
        top_k=[40],
        repetitions=2,
        max_tokens=64,
        timeout_seconds=5,
        dataset_path=ds,
        eval_split="test",
        output_path=tmp_path / "out.jsonl",
        judges=[{"type": "heuristic"}],
        trial_limit=None,
    )

    trials = _make_trials(cfg)

    assert len(trials) == 4
    assert {t.item_id for t in trials} == {"a1"}


def test_run_benchmark_with_mocks_writes_jsonl(tmp_path, monkeypatch):
    ds = tmp_path / "dataset.jsonl"
    _write_dataset(ds)

    cfg = BenchmarkConfig(
        random_seed=42,
        base_url="http://localhost:11434",
        models=["m1"],
        temperatures=[0.2],
        top_p=[0.9],
        top_k=[40],
        repetitions=1,
        max_tokens=64,
        timeout_seconds=5,
        dataset_path=ds,
        eval_split="test",
        output_path=tmp_path / "out.jsonl",
        judges=[{"type": "heuristic"}],
        trial_limit=1,
    )

    monkeypatch.setattr("slm_benchmark.runner.OllamaClient", _FakeClient)
    monkeypatch.setattr("slm_benchmark.runner.build_judges", lambda cfg, client: [_FakeJudge()])

    out_path = run_benchmark(cfg)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["valid_response"] is True
    assert payload["judge_scores"]["fake_judge"] == 8.0
    assert payload["score_aggregated"] == 8.0


def test_ollama_judge_fallback_parsing_works():
    class _JudgeClient:
        def generate(self, **kwargs):
            return GenerationOutput(text="score: 7.5/10 rationale: ok", latency_ms=5)

    judge = OllamaJudge(client=_JudgeClient(), model="judge-model")
    scored = judge.score("Resposta do deepseek-coder:6.7b", "ref")

    assert scored.score == 7.5
    assert scored.rationale.startswith("judge_parse_fallback_used:")


def test_ollama_judge_handles_multiple_brace_blocks_with_fallback():
    class _JudgeClient:
        def generate(self, **kwargs):
            return GenerationOutput(text="noise {bad} score: 6.0/10 then {also_bad}", latency_ms=5)

    judge = OllamaJudge(client=_JudgeClient(), model="judge-model")
    scored = judge.score("answer", "ref")

    assert scored.score == 6.0
    assert scored.rationale.startswith("judge_parse_fallback_used:")


def test_run_benchmark_marks_invalid_when_all_judges_fail(tmp_path, monkeypatch):
    ds = tmp_path / "dataset.jsonl"
    _write_dataset(ds)

    cfg = BenchmarkConfig(
        random_seed=42,
        base_url="http://localhost:11434",
        models=["m1"],
        temperatures=[0.2],
        top_p=[0.9],
        top_k=[40],
        repetitions=1,
        max_tokens=64,
        timeout_seconds=5,
        dataset_path=ds,
        eval_split="test",
        output_path=tmp_path / "out.jsonl",
        judges=[{"type": "heuristic"}],
        trial_limit=1,
    )

    monkeypatch.setattr("slm_benchmark.runner.OllamaClient", _FakeClient)
    monkeypatch.setattr("slm_benchmark.runner.build_judges", lambda cfg, client: [_FakeNoScoreJudge()])

    out_path = run_benchmark(cfg)
    payload = json.loads(out_path.read_text(encoding="utf-8").strip())

    assert payload["judge_valid_count"] == 0
    assert payload["score_aggregated"] is None
    assert payload["valid_response"] is False


def test_ollama_client_returns_error_for_unexpected_payload(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"unexpected": {"content": "x"}}

    monkeypatch.setattr("slm_benchmark.clients.requests.post", lambda *args, **kwargs: _Resp())

    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1)
    out = client.generate(
        model="m1",
        prompt="p",
        temperature=0.0,
        top_p=0.9,
        top_k=40,
        max_tokens=16,
    )

    assert out.text == ""
    assert out.error is not None
    assert "Unexpected Ollama response" in out.error
