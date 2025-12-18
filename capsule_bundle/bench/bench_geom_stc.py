#!/usr/bin/env python3
"""Run the geometry STC+FRI demo across a grid of parameters and record profiles."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "zk_geom_demo.py")
PYTHON = sys.executable


def run_once(steps: int, num_queries: int) -> dict:
    cmd = [
        PYTHON,
        SCRIPT,
        "prove",
        f"--steps={steps}",
        f"--num-queries={num_queries}",
        "--profile",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = proc.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError("zk_geom_demo produced no output")
    try:
        profile = json.loads(stdout[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Expected JSON profile on last line; got: " + stdout[-1]
        ) from exc
    profile.setdefault("backend", "geom_stc_fri")
    profile.setdefault("steps", steps)
    profile.setdefault("num_queries", num_queries)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark geometry STC+FRI prover")
    parser.add_argument("--min-log2", type=int, default=8, help="min log2(trace length)")
    parser.add_argument("--max-log2", type=int, default=12, help="max log2(trace length)")
    parser.add_argument(
        "--queries", type=int, nargs="+", default=[8, 16], help="query counts to test"
    )
    parser.add_argument("--output", type=str, help="optional JSON output path")
    args = parser.parse_args()

    results: List[dict] = []
    for log2 in range(args.min_log2, args.max_log2 + 1):
        steps = 1 << log2
        for q in args.queries:
            profile = run_once(steps, q)
            results.append(profile)
            print(
                f"steps={steps} queries={q} proof={profile['proof_size_bytes']}B "
                f"prove={profile['proving_time_sec']:.3f}s verify={profile['verify_time_sec']:.4f}s"
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
