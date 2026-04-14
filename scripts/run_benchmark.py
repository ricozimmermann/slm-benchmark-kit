from __future__ import annotations

import argparse

from slm_benchmark.clients import OllamaClient
from slm_benchmark.config import load_config
from slm_benchmark.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SLM benchmark trials")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Check Ollama connectivity and configured models before running",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.check_local:
        client = OllamaClient(base_url=cfg.base_url, timeout_seconds=cfg.timeout_seconds)
        ok, msg = client.healthcheck()
        if not ok:
            raise SystemExit(f"Local Ollama check failed ({cfg.base_url}): {msg}")

        available_models = set(client.list_models())
        missing = [m for m in cfg.models if m not in available_models]
        print(f"Local Ollama reachable at: {cfg.base_url}")
        print(f"Detected models: {len(available_models)}")
        if missing:
            print(f"Warning: configured models not found locally: {missing}")

    output = run_benchmark(cfg)
    print(f"Saved raw results to: {output}")


if __name__ == "__main__":
    main()
