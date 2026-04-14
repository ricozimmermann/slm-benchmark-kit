from __future__ import annotations

import argparse

from slm_benchmark.agreement import agreement_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate agreement report from human eval scores")
    parser.add_argument("--input", required=True, help="Scored assignment CSV")
    parser.add_argument("--output", required=True, help="Output markdown report")
    args = parser.parse_args()

    pair_df = agreement_markdown_report(args.input, args.output)
    print(f"Agreement pairs: {len(pair_df)}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
