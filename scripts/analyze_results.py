from __future__ import annotations

import argparse
from pathlib import Path

from slm_benchmark.analysis import generate_markdown_report, load_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SLM benchmark results")
    parser.add_argument("--input", required=True, help="Input JSONL results file")
    parser.add_argument("--output", required=True, help="Output Markdown report path")
    args = parser.parse_args()

    df = load_results(args.input)
    generate_markdown_report(df, Path(args.output))
    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
