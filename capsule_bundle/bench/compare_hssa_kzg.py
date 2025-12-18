#!/usr/bin/env python3
"""Merge HSSA and KZG benchmark CSVs into a combined comparison."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_csv(path: Path, key_fields):
    data = {}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = tuple(int(row[field]) for field in key_fields)
            data[key] = row
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine HSSA and KZG benchmark CSVs")
    parser.add_argument("--hssa", type=Path, default=Path("bench/hssa_results.csv"), help="Path to HSSA CSV")
    parser.add_argument("--kzg", type=Path, default=Path("bench/kzg_results.csv"), help="Path to KZG CSV")
    parser.add_argument("--output", type=Path, default=Path("bench/combined_results.csv"), help="Output CSV")
    parser.add_argument("--target-m", type=int, default=4, help="Filter HSSA rows by m")
    parser.add_argument("--target-chunk", type=int, default=8192, help="Filter HSSA rows by chunk length")
    args = parser.parse_args()

    hssa_rows = load_csv(args.hssa, key_fields=["N", "m", "chunk_len"])
    kzg_rows = load_csv(args.kzg, key_fields=["N"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        fieldnames = ["N", "m", "chunk_len", "hssa_throughput_gb_s", "kzg_throughput_mb_s", "speed_ratio"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, hssa_row in hssa_rows.items():
            N, m, chunk_len = key
            if m != args.target_m or chunk_len != args.target_chunk:
                continue
            kzg_row = kzg_rows.get((N,))
            if not kzg_row:
                continue
            hssa_throughput = float(hssa_row.get("throughput_gbps", 0.0))
            kzg_throughput_mb = float(kzg_row.get("throughput_mb_s", 0.0))
            kzg_throughput_gb = kzg_throughput_mb / 1024.0
            speed_ratio = hssa_throughput / kzg_throughput_gb if kzg_throughput_gb > 0 else 0.0
            writer.writerow(
                {
                    "N": N,
                    "m": m,
                    "chunk_len": chunk_len,
                    "hssa_throughput_gb_s": f"{hssa_throughput:.6f}",
                    "kzg_throughput_mb_s": f"{kzg_throughput_mb:.6f}",
                    "speed_ratio": f"{speed_ratio:.2f}",
                }
            )
    print(f"Combined results written to {args.output}")


if __name__ == "__main__":
    main()
