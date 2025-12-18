#!/usr/bin/env python3
"""Compare no-zk vs zk frontends for the geometry demo."""
from __future__ import annotations

import argparse
import json
import subprocess

SCRIPT = "scripts/zk_geom_demo.py"
PYTHON = "python"


def run(cmd):
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-log2", type=int, default=10)
    parser.add_argument("--max-log2", type=int, default=12)
    parser.add_argument("--queries", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--output", type=str, default="geom_compare.json")
    args = parser.parse_args()

    results = []
    for log2 in range(args.min_log2, args.max_log2 + 1):
        steps = 1 << log2
        baseline = run([PYTHON, SCRIPT, "prove", f"--steps", str(steps), "--no-zk"])
        results.append(baseline)
        for q in args.queries:
            res = run([
                PYTHON,
                SCRIPT,
                "prove",
                f"--steps",
                str(steps),
                "--num-queries",
                str(q),
                "--profile",
            ])
            results.append(res)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
