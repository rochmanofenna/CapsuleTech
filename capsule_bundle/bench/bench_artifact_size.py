#!/usr/bin/env python3
"""Benchmark artifact sizes for a given configuration."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    repo = str(REPO_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo if not existing else f"{repo}:{existing}"
    return env


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact size benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("out/bench_size"))
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--proof-threshold", type=float, default=1.75)
    parser.add_argument("--capsule-threshold", type=float, default=1.6)
    parser.add_argument("--encoding-id", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_extra = []
    if args.encoding_id:
        cmd_extra += ["--encoding-id", args.encoding_id]
    env = _python_env()
    pipeline_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_geom_pipeline.py"),
        "--steps",
        str(args.steps),
        "--num-queries",
        str(args.num_queries),
        "--output-dir",
        str(out_dir),
        "--skip-nova",
    ] + cmd_extra
    subprocess.run(pipeline_cmd, check=True, cwd=REPO_ROOT, env=env)

    proof_json = out_dir / "geom_proof.json"
    proof_bin = out_dir / "geom_proof.bin"
    capsule_json = out_dir / "strategy_capsule.json"
    capsule_bin = out_dir / "strategy_capsule.bin"

    proof_json_size = _size(proof_json)
    proof_bin_size = _size(proof_bin)
    capsule_json_size = _size(capsule_json)
    capsule_bin_size = _size(capsule_bin)

    if proof_bin_size == 0 or capsule_bin_size == 0:
        raise SystemExit("missing binary artifacts; rerun pipeline with --artifact-formats bin/both")

    proof_ratio = proof_json_size / proof_bin_size
    capsule_ratio = capsule_json_size / capsule_bin_size

    print("Artifact size report:")
    print(f"  Proof JSON:    {proof_json_size:,} bytes")
    print(f"  Proof BIN:     {proof_bin_size:,} bytes (ratio {proof_ratio:.2f}x)")
    print(f"  Capsule JSON:  {capsule_json_size:,} bytes")
    print(f"  Capsule BIN:   {capsule_bin_size:,} bytes (ratio {capsule_ratio:.2f}x)")

    if proof_ratio < args.proof_threshold:
        raise RuntimeError(
            f"proof shrink ratio {proof_ratio:.2f}x below threshold {args.proof_threshold:.2f}x"
        )
    if capsule_ratio < args.capsule_threshold:
        raise RuntimeError(
            f"capsule shrink ratio {capsule_ratio:.2f}x below threshold {args.capsule_threshold:.2f}x"
        )
    print("Benchmarks satisfied shrink targets.")


if __name__ == "__main__":
    main()
