from __future__ import annotations

import argparse

from slm_benchmark.config import load_config
from slm_benchmark.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SLM benchmark trials")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = run_benchmark(cfg)
    print(f"Saved raw results to: {output}")


if __name__ == "__main__":
    main()
