from __future__ import annotations

import argparse

from slm_benchmark.human_eval import SamplingPlan, prepare_blind_human_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare blind human evaluation assignments")
    parser.add_argument("--input", required=True, help="Raw benchmark JSONL")
    parser.add_argument("--assignment", required=True, help="Output assignment CSV for annotators")
    parser.add_argument("--key", required=True, help="Output decode key CSV (keep private)")
    parser.add_argument("--evaluators", required=True, nargs="+", help="Evaluator IDs")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap-rate", type=float, default=0.25)
    args = parser.parse_args()

    plan = SamplingPlan(sample_size=args.sample_size, seed=args.seed, overlap_rate=args.overlap_rate)
    assignment_df, key_df = prepare_blind_human_eval(
        raw_results_path=args.input,
        out_assignment_csv=args.assignment,
        out_key_csv=args.key,
        evaluators=args.evaluators,
        plan=plan,
    )

    print(f"Assignments: {len(assignment_df)} rows -> {args.assignment}")
    print(f"Decode key: {len(key_df)} rows -> {args.key}")


if __name__ == "__main__":
    main()
