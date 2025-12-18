#!/usr/bin/env python3
"""Benchmark HSSA (StreamingAccumulatorCUDA + bef_verify_fast)."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu_accumulator.stream_accumulator import DEFAULT_MODULUS, StreamingAccumulatorCUDA  # type: ignore

DEFAULT_CLI = REPO_ROOT / "BICEPsrc" / "BICEPrust" / "bicep" / "target" / "release" / "bef_verify_fast_cli"


def parse_list(arg: str) -> List[int]:
    return [int(x.strip(), 0) for x in arg.split(",") if x.strip()]


def build_sketch_dict(proof: Dict, modulus: int, trace_id: str, seed: int) -> Dict:
    timing = {}
    for key, dst in (
        ("cuda_rpow_ms", "cuda_rpow"),
        ("cuda_chunks_ms", "cuda_chunks"),
        ("cuda_fused_global_ms", "cuda_fused_global"),
        ("cuda_fused_chunks_ms", "cuda_fused_chunks"),
    ):
        value = proof.get(key)
        if value is not None:
            timing[dst] = value
    sketch = {
        "schema": "bef_sketch_v1",
        "trace_id": trace_id,
        "field_modulus": modulus,
        "seed": seed,
        "length": proof.get("length"),
        "challenge": proof.get("r"),
        "global_sketch": proof.get("global_sketch"),
        "challenges": proof.get("challenges", []),
        "global_sketch_vec": proof.get("global_sketch_vec", []),
        "trace_commitment": proof.get("trace_commitment"),
        "commitment_root": proof.get("commitment_root"),
        "chunks": [
            {
                "chunk_index": idx,
                "offset": chunk["offset"],
                "length": chunk["length"],
                "root_hex": chunk["root"],
                "sketch_vec": chunk["sketch_vec"],
            }
            for idx, chunk in enumerate(proof.get("chunks", []))
        ],
    }
    if timing:
        sketch["timing_ms"] = timing
    return sketch


def generate_chunks(total_len: int, chunk_len: int, modulus: int, seed: int) -> Iterable[List[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    produced = 0
    while produced < total_len:
        size = min(chunk_len, total_len - produced)
        values = torch.randint(0, modulus, (size,), dtype=torch.int64, generator=generator)
        yield values.tolist()
        produced += size


def run_verify_cli(cli_path: Path, sketch_path: Path, timeout: float) -> float:
    start = time.perf_counter()
    result = subprocess.run(
        [str(cli_path), str(sketch_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bef_verify_fast_cli failed: {result.stderr.strip()}" )
    stdout = result.stdout.strip()
    try:
        return float(stdout)
    except ValueError:
        return (time.perf_counter() - start) * 1_000.0


def run_single_bench(
    N: int,
    m: int,
    chunk_len: int,
    seed: int,
    verify_cli: Optional[Path],
    tmp_dir: Path,
    verify_timeout: float,
) -> Dict:
    acc = StreamingAccumulatorCUDA(num_challenges=m, seed=seed, use_fused=True)
    for chunk_values in generate_chunks(N, chunk_len, DEFAULT_MODULUS, seed):
        acc.add_chunk(chunk_values)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    proof = acc.prove()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    commit_wall_ms = (t1 - t0) * 1_000.0

    sketch = build_sketch_dict(proof, DEFAULT_MODULUS, f"bench_N{N}_m{m}", seed)
    sketch_path = tmp_dir / f"sketch_N{N}_m{m}_seed{seed}.json"
    sketch_path.write_text(json.dumps(sketch))

    verify_ms = None
    if verify_cli is not None:
        verify_ms = run_verify_cli(verify_cli, sketch_path, verify_timeout)

    num_chunks = len(proof.get("chunks", []))
    fused_global = proof.get("cuda_fused_global_ms")
    fused_chunks = proof.get("cuda_fused_chunks_ms")
    throughput_gbps = 0.0
    if commit_wall_ms > 0:
        bytes_processed = N * 8
        throughput_gbps = bytes_processed / (commit_wall_ms / 1_000.0) / (1024 ** 3)

    del acc
    torch.cuda.empty_cache()

    return {
        "N": N,
        "m": m,
        "chunk_len": chunk_len,
        "seed": seed,
        "num_chunks": num_chunks,
        "commit_wall_ms": commit_wall_ms,
        "cuda_fused_global_ms": fused_global,
        "cuda_fused_chunks_ms": fused_chunks,
        "verify_ms": verify_ms,
        "throughput_gbps": throughput_gbps,
        "sketch_path": str(sketch_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HSSA (GPU + bef_verify_fast).")
    parser.add_argument("--Ns", default="1048576,16777216", help="Comma-separated N values (field elements)")
    parser.add_argument("--challenges", default="2,4,8", help="Comma-separated m values")
    parser.add_argument("--chunk-lens", default="1024,8192", help="Comma-separated chunk lengths")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per configuration")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "bench" / "hssa_results.csv")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV")
    parser.add_argument("--verify-cli", type=Path, default=DEFAULT_CLI, help="Path to bef_verify_fast_cli binary")
    parser.add_argument("--skip-verify", action="store_true", help="Skip bef_verify_fast timing")
    parser.add_argument("--verify-timeout", type=float, default=60.0, help="Seconds per verify CLI call")
    parser.add_argument("--tmp-dir", type=Path, help="Directory for temporary sketches (defaults to tmp)")
    parser.add_argument("--base-seed", type=int, default=1234, help="Base RNG seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for HSSA benchmark")

    Ns = parse_list(args.Ns)
    m_values = parse_list(args.challenges)
    chunk_lens = parse_list(args.chunk_lens)

    verify_cli: Optional[Path]
    if args.skip_verify:
        verify_cli = None
    else:
        verify_cli = args.verify_cli
        if not verify_cli.exists():
            raise FileNotFoundError(f"verify CLI not found at {verify_cli}. Build it via `cargo build --release -p bicep-crypto --bin bef_verify_fast_cli`." )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.output.exists() else "w"
    write_header = mode == "w"

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_root:
        tmp_dir = Path(tmp_root)
        with args.output.open(mode, newline="") as csvfile:
            fieldnames = [
                "N",
                "m",
                "chunk_len",
                "repeat",
                "seed",
                "num_chunks",
                "commit_wall_ms",
                "cuda_fused_global_ms",
                "cuda_fused_chunks_ms",
                "verify_ms",
                "throughput_gbps",
                "sketch_path",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            total_configs = len(Ns) * len(m_values) * len(chunk_lens)
            config_idx = 0
            for N in Ns:
                for m in m_values:
                    for chunk_len in chunk_lens:
                        config_idx += 1
                        print(f"[{config_idx}/{total_configs}] N={N} m={m} chunk_len={chunk_len}")
                        for repeat in range(args.repeats):
                            seed = args.base_seed + repeat
                            result = run_single_bench(
                                N=N,
                                m=m,
                                chunk_len=chunk_len,
                                seed=seed,
                                verify_cli=verify_cli,
                                tmp_dir=tmp_dir,
                                verify_timeout=args.verify_timeout,
                            )
                            result["repeat"] = repeat
                            writer.writerow(result)
                            csvfile.flush()
                            torch.cuda.empty_cache()
    print(f"Benchmark complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
