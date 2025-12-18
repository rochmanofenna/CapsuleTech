#!/usr/bin/env python3
"""Convert geom_bench.json records into a compact CSV summary."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, Iterable


def _load_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("expected list of records in JSON benchmark file")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Export geom_bench JSON to CSV")
    parser.add_argument("--input", default="geom_bench.json", help="path to geom_bench.json")
    parser.add_argument("--output", help="optional CSV output path; defaults to stdout")
    args = parser.parse_args()

    records = _load_records(args.input)
    fieldnames = [
        "backend",
        "steps",
        "num_queries",
        "proof_size_bytes",
        "proving_time_sec",
        "verify_time_sec",
        "trace_time_sec",
        "overhead_zk_vs_trace",
    ]

    out_fh = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda r: (r.get("backend", ""), int(r.get("steps", 0)), int(r.get("num_queries", 0))),
        ):
            trace_time = float(record.get("trace_time_total_sec", 0.0)) or float(
                record.get("trace_time_sec", 0.0)
            )
            proving_time = float(record.get("proving_time_sec", 0.0))
            overhead = proving_time / trace_time if trace_time else ""
            writer.writerow(
                {
                    "backend": record.get("backend", "geom_stc_fri"),
                    "steps": int(record.get("steps", 0)),
                    "num_queries": int(record.get("num_queries", 0)),
                    "proof_size_bytes": int(record.get("proof_size_bytes", 0)),
                    "proving_time_sec": proving_time,
                    "verify_time_sec": float(record.get("verify_time_sec", 0.0)),
                    "trace_time_sec": trace_time,
                    "overhead_zk_vs_trace": overhead,
                }
            )
    finally:
        if args.output:
            out_fh.close()


if __name__ == "__main__":
    main()
