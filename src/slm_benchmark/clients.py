from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class GenerationOutput:
    text: str
    latency_ms: int
    error: str | None = None


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def healthcheck(self) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
            resp.raise_for_status()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        names = []
        for m in models:
            name = m.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> GenerationOutput:
        import time

        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "num_predict": max_tokens,
                    },
                    "stream": False,
                },
                timeout=self.timeout_seconds,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            resp.raise_for_status()
            data = resp.json()
            return GenerationOutput(text=data.get("message", {}).get("content", ""), latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return GenerationOutput(text="", latency_ms=latency_ms, error=str(exc))
