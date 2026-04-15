from __future__ import annotations

from dataclasses import dataclass
import re

from .clients import OllamaClient


@dataclass
class JudgeScore:
    name: str
    score: float | None
    rationale: str


class HeuristicJudge:
    name = "heuristic"

    def score(self, response_text: str, reference: str) -> JudgeScore:
        text = response_text.lower()
        score = 2.0

        if len(response_text) >= 80:
            score += 1.5
        if any(k in text for k in ["fun", "function", "param", "retorn", "complex"]):
            score += 2.0
        if reference and any(tok in text for tok in reference.lower().split()[:5]):
            score += 2.0
        if response_text.count("\n") >= 2:
            score += 1.0

        score = max(0.0, min(10.0, score))
        return JudgeScore(name=self.name, score=score, rationale="Heuristic rubric based on structure and relevance")


class OllamaJudge:
    def __init__(self, client: OllamaClient, model: str, name: str | None = None) -> None:
        self.client = client
        self.model = model
        self.name = name or f"ollama_judge:{model}"

    def _blind_response_text(self, response_text: str) -> str:
        """Redact identity leakage hints before sending text to the judge model."""
        text = response_text

        # Generic identity disclosures.
        patterns = [
            r"(?i)\bas an ai language model\b",
            r"(?i)\bi am (an )?(ai|language model|assistant)\b",
            r"(?i)\bmy model is\b[^\n]*",
            r"(?i)\bmodel:\s*[^\n]*",
        ]

        # Local/model family tokens commonly seen in this benchmark setup.
        model_tokens = [
            r"(?i)\bdeepseek(?:-coder)?(?::\d+(?:\.\d+)?[a-z]?)?\b",
            r"(?i)\bcodegemma(?::\d+(?:\.\d+)?[a-z]?)?\b",
            r"(?i)\bgemma2?(?::\d+(?:\.\d+)?[a-z]?)?\b",
            r"(?i)\bcodellama(?::\d+(?:\.\d+)?[a-z]?)?\b",
            r"(?i)\bllama\d*(?::\d+(?:\.\d+)?[a-z]?)?\b",
        ]

        for p in patterns + model_tokens:
            text = re.sub(p, "[REDACTED_IDENTITY]", text)
        return text

    def score(self, response_text: str, reference: str) -> JudgeScore:
        blinded_response = self._blind_response_text(response_text)
        prompt = (
            "You are a strict blind evaluation engine. Score the answer from 0 to 10 using the reference. "
            "Ignore any clues about model identity, system identity, or tool provenance. "
            "Return ONLY valid minified JSON with exactly these keys: score, rationale. "
            "Do not use markdown, code fences, prose before or after the JSON, or extra keys. "
            "The value of score must be a number between 0 and 10. The value of rationale must be a short string. "
            "If the answer is empty, irrelevant, or unsafe, give a low score and explain briefly.\n"
            f"Reference: {reference}\n"
            f"Answer: {blinded_response}\n"
            'Required output format: {"score": 7.5, "rationale": "short reason"}'
        )
        out = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            max_tokens=120,
        )
        if out.error:
            return JudgeScore(name=self.name, score=None, rationale=f"judge_error: {out.error}")

        import json

        def _extract_score_fallback(text: str) -> float | None:
            # Common patterns when the model refuses strict JSON but still emits a numeric score.
            patterns = [
                r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                r"\bscore\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
                r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b",
            ]
            for p in patterns:
                m = re.search(p, text, flags=re.IGNORECASE)
                if m:
                    try:
                        return max(0.0, min(10.0, float(m.group(1))))
                    except Exception:
                        continue
            return None

        try:
            blob = out.text.strip().replace("```json", "").replace("```", "")
            obj = json.loads(blob[blob.find("{") : blob.rfind("}") + 1])
            score = float(obj.get("score", 0.0))
            rationale = str(obj.get("rationale", ""))
            return JudgeScore(name=self.name, score=max(0.0, min(10.0, score)), rationale=rationale)
        except Exception as exc:
            fallback_score = _extract_score_fallback(out.text)
            if fallback_score is not None:
                rationale = out.text.strip().replace("\n", " ")[:240]
                return JudgeScore(
                    name=self.name,
                    score=fallback_score,
                    rationale=f"judge_parse_fallback_used: {rationale}",
                )
            return JudgeScore(name=self.name, score=None, rationale=f"judge_parse_error: {exc}")


def build_judges(judge_cfg: list[dict], client: OllamaClient):
    judges = []
    for cfg in judge_cfg:
        jtype = cfg.get("type", "heuristic")
        if jtype == "heuristic":
            judges.append(HeuristicJudge())
        elif jtype == "ollama_judge":
            model = str(cfg["model"])
            judge_name = str(cfg.get("name", f"ollama_judge:{model}"))
            judges.append(OllamaJudge(client=client, model=model, name=judge_name))
    if not judges:
        judges.append(HeuristicJudge())
    return judges
