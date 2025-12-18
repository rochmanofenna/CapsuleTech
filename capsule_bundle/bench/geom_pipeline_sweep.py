#!/usr/bin/env python3
"""Sweep the geometry pipeline across parameters and collect timings."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_geom_pipeline.py"
PYTHON = Path(sys.executable)


def run_pipeline_once(
    *,
    steps: int,
    num_challenges: int,
    num_queries: int,
    compressed: bool,
    trace_id: str,
    tmp_root: Path,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"geom_{steps}_", dir=tmp_root) as tmpdir:
        tmp_path = Path(tmpdir)
        stats_path = tmp_path / "pipeline_stats.json"
        cmd = [
            str(PYTHON),
            str(PIPELINE),
            f"--steps={steps}",
            f"--num-challenges={num_challenges}",
            f"--num-queries={num_queries}",
            f"--output-dir={tmp_path}",
            f"--trace-id={trace_id}",
            f"--stats-out={stats_path}",
        ]
        if not compressed:
            cmd.append("--no-compressed")
        subprocess.run(cmd, check=True, cwd=ROOT)
        with stats_path.open("r", encoding="utf-8") as fh:
            stats = json.load(fh)
    return stats


def summarize(stats: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "steps": stats["steps"],
        "num_challenges": stats["num_challenges"],
        "num_queries": stats["num_queries"],
        "row_width": stats["row_width"],
        "trace_time_ms": stats["trace_time_sec"] * 1e3,
        "geom_proof_bytes": stats["geom_proof"]["size_bytes"],
    }
    geom_profile = stats["geom_proof"].get("profile", {})
    record.update(
        {
            "geom_time_total_ms": geom_profile.get("time_total_sec", 0.0) * 1e3,
            "geom_trace_ms": geom_profile.get("time_trace_sec", 0.0) * 1e3,
            "geom_row_commit_ms": geom_profile.get("time_row_commit_sec", 0.0) * 1e3,
            "geom_pc_commit_ms": geom_profile.get("time_pc_commit_sec", 0.0) * 1e3,
            "geom_fri_ms": geom_profile.get("time_fri_sec", 0.0) * 1e3,
            "geom_row_openings": geom_profile.get("row_openings"),
            "geom_backend": geom_profile.get("backend"),
        }
    )
    nova = stats["nova"]
    timings = nova.get("timings_ms", {})
    record.update(
        {
            "nova_plain_ms": timings.get("plain"),
            "nova_prove_total_ms": timings.get("prove_total"),
            "nova_verify_ms": timings.get("verify"),
            "nova_prove_avg_ms": timings.get("prove_avg"),
            "nova_pp_ms": timings.get("pp"),
            "nova_base_ms": timings.get("base"),
            "nova_recursive_bytes": nova.get("recursive_proof_bytes"),
            "nova_constraints_primary": nova.get("constraints", {}).get("primary"),
            "nova_constraints_secondary": nova.get("constraints", {}).get("secondary"),
        }
    )
    compressed = nova.get("compressed")
    if compressed:
        record.update(
            {
                "compressed_setup_ms": compressed.get("setup"),
                "compressed_prove_ms": compressed.get("prove"),
                "compressed_verify_ms": compressed.get("verify"),
                "compressed_proof_bytes": compressed.get("proof_bytes"),
            }
        )
    else:
        record.update(
            {
                "compressed_setup_ms": None,
                "compressed_prove_ms": None,
                "compressed_verify_ms": None,
                "compressed_proof_bytes": None,
            }
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the geom pipeline")
    parser.add_argument("--log2-min", type=int, default=6, help="min log2(trace length)")
    parser.add_argument("--log2-max", type=int, default=10, help="max log2(trace length)")
    parser.add_argument(
        "--challenges",
        type=int,
        nargs="+",
        default=[2],
        help="challenge counts to test",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        default=[8],
        help="FRI query counts to test",
    )
    parser.add_argument("--repeats", type=int, default=1, help="repetitions per configuration")
    parser.add_argument(
        "--no-compressed",
        action="store_true",
        help="skip compressed SNARK stage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "geom_pipeline_sweep.json",
        help="JSON file for collected results",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=ROOT / "out" / "pipeline_sweep_tmp",
        help="directory for temporary per-run artifacts",
    )
    args = parser.parse_args()

    args.tmp_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    run_id = 0
    for log2_steps in range(args.log2_min, args.log2_max + 1):
        steps = 1 << log2_steps
        for m in args.challenges:
            for q in args.queries:
                for rep in range(args.repeats):
                    run_id += 1
                    trace_id = f"sweep_{steps}_m{m}_q{q}_r{rep}"
                    print(
                        f"[run {run_id}] steps={steps} m={m} q={q} rep={rep+1}/{args.repeats}"
                    )
                    stats = run_pipeline_once(
                        steps=steps,
                        num_challenges=m,
                        num_queries=q,
                        compressed=not args.no_compressed,
                        trace_id=trace_id,
                        tmp_root=args.tmp_root,
                    )
                    record = summarize(stats)
                    results.append(record)
                    print(
                        f"    geom_total={record['geom_time_total_ms']:.2f} ms, "
                        f"nova_prove={record['nova_prove_total_ms']:.2f} ms, "
                        f"compressed_prove={record['compressed_prove_ms']} ms"
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
