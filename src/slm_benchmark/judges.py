from __future__ import annotations

from dataclasses import dataclass

from .clients import OllamaClient


@dataclass
class JudgeScore:
    name: str
    score: float
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
    name = "ollama_judge"

    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client = client
        self.model = model

    def score(self, response_text: str, reference: str) -> JudgeScore:
        prompt = (
            "Avalie a resposta de 0 a 10. Retorne SOMENTE JSON: "
            '{"score": <float>, "rationale": "<texto>"}. '
            f"Referencia: {reference}\nResposta: {response_text}"
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
            return JudgeScore(name=self.name, score=0.0, rationale=f"judge_error: {out.error}")

        import json

        try:
            blob = out.text.strip().replace("```json", "").replace("```", "")
            obj = json.loads(blob[blob.find("{") : blob.rfind("}") + 1])
            score = float(obj.get("score", 0.0))
            rationale = str(obj.get("rationale", ""))
            return JudgeScore(name=self.name, score=max(0.0, min(10.0, score)), rationale=rationale)
        except Exception as exc:
            return JudgeScore(name=self.name, score=0.0, rationale=f"judge_parse_error: {exc}")


def build_judges(judge_cfg: list[dict], client: OllamaClient):
    judges = []
    for cfg in judge_cfg:
        jtype = cfg.get("type", "heuristic")
        if jtype == "heuristic":
            judges.append(HeuristicJudge())
        elif jtype == "ollama_judge":
            judges.append(OllamaJudge(client=client, model=str(cfg["model"])))
    if not judges:
        judges.append(HeuristicJudge())
    return judges
