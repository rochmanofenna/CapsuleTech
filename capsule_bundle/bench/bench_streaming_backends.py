#!/usr/bin/env python3
"""Benchmark streaming row backends (Merkle vs STC) on GeomAIR traces."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from bef_zk.air.geom_air import (
    GeomAIRParams,
    GeomInitialState,
    next_power_of_two,
    simulate_trace,
    trace_to_eval_table,
)
from bef_zk.zk_geom.backend import available_row_backends, get_row_backend
from bef_zk.zk_geom.columns import build_row_matrix, extract_masked_columns
from bef_zk.zk_geom.masking import derive_column_masks
from bef_zk.zk_geom.prover import _derive_alpha_digest

PROGRAM = [1, 2, 3, 1, 2, 0]


def build_masked_rows(steps: int) -> tuple[List[List[int]], GeomAIRParams]:
    params = GeomAIRParams(
        steps=steps,
        num_challenges=2,
        r_challenges=[1234567, 89101112],
        matrix=[[2, 1], [1, 1]],
    )
    init = GeomInitialState()
    trace = simulate_trace(PROGRAM, params, init)
    domain_size = next_power_of_two(steps)
    alpha_digest = _derive_alpha_digest(params)
    column_masks = derive_column_masks(alpha_digest, params, domain_size)
    eval_table = trace_to_eval_table(trace, domain_size, column_masks=column_masks)
    masked_columns = extract_masked_columns(eval_table, params)
    rows = build_row_matrix(masked_columns, params)
    return rows, params


def run_backend_stream(rows: List[List[int]], backend_name: str, epoch_size: int) -> List[dict]:
    if not rows:
        return []
    row_width = len(rows[0])
    backend = get_row_backend(backend_name, row_width)
    stats: List[dict] = []
    idx = 0
    epoch = 0
    total_rows = len(rows)
    chunk_size = epoch_size if epoch_size > 0 else total_rows
    while idx < total_rows:
        chunk = rows[idx : min(idx + chunk_size, total_rows)]
        state = backend.streaming_init()
        append_cost = 0.0
        for row in chunk:
            t0 = time.perf_counter()
            backend.streaming_append(state, row)
            append_cost += time.perf_counter() - t0
        t0 = time.perf_counter()
        backend.streaming_finalize(state)
        finalize_cost = time.perf_counter() - t0
        stats.append(
            {
                "backend": backend_name,
                "epoch_index": epoch,
                "rows": len(chunk),
                "stream_append_total_sec": append_cost,
                "stream_append_avg_sec": (append_cost / len(chunk)) if chunk else 0.0,
                "stream_finalize_sec": finalize_cost,
            }
        )
        idx += len(chunk)
        epoch += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream row backends and measure append/finalize cost")
    parser.add_argument("--steps", type=int, default=4096, help="number of trace steps to simulate")
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=4096,
        help="rows per streaming epoch before finalizing (defaults to steps)",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=available_row_backends(),
        help="backends to benchmark (default: both)",
    )
    parser.add_argument("--output", type=str, help="optional path to write JSON results")
    args = parser.parse_args()

    rows, _ = build_masked_rows(args.steps)
    backends = args.backend if args.backend else available_row_backends()
    results: List[dict] = []
    for backend_name in backends:
        results.extend(run_backend_stream(rows, backend_name, args.epoch_size))

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
