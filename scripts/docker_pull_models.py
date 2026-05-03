from __future__ import annotations

import os
import time

import requests


def _wait_ollama(base_url: str, timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "unknown"
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"Ollama not ready after {timeout_seconds}s: {last_error}")


def _pull_model(base_url: str, model: str) -> None:
    resp = requests.post(
        f"{base_url}/api/pull",
        json={"name": model, "stream": False},
        timeout=1800,
    )
    resp.raise_for_status()


def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
    models_csv = os.getenv(
        "OLLAMA_MODELS",
        "deepseek-coder:1.3b,qwen2.5-coder:1.5b,gemma2:2b,codellama:7b",
    )
    models = [m.strip() for m in models_csv.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models provided in OLLAMA_MODELS")

    _wait_ollama(base_url)

    for model in models:
        print(f"Pulling model: {model}")
        _pull_model(base_url, model)

    print(f"Done. Pulled {len(models)} model(s).")


if __name__ == "__main__":
    main()
