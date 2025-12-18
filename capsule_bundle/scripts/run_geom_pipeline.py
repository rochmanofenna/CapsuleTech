#!/usr/bin/env python3
"""Run the full Geometry → STC → Nova demo end-to-end."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState, simulate_trace
from bef_zk.air.geom_trace_export import geom_trace_to_bef_trace, flatten_geom_row
from bef_zk.codec import (
    ENCODING_ID,
    canonical_encode,
    compute_capsule_hash,
)
from bef_zk.fri.config import FRIConfig
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.zk_geom.prover import zk_prove_geom
from bef_zk.zk_geom.serialization import (
    proof_from_json,
    proof_to_json,
    proof_to_bytes,
)
from bef_zk.zk_geom.verifier import zk_verify_geom
from bef_zk.zk_geom.columns import column_names
from bef_zk.spec import (
    TraceSpecV1,
    StatementV1,
    compute_trace_spec_hash,
    compute_statement_hash,
)
from scripts.geom_programs import GEOM_PROGRAM

try:
    from coincurve import PrivateKey
except ImportError:  # pragma: no cover - optional dependency
    PrivateKey = None

ROOT = Path(__file__).parent.parent.resolve()
MODULUS = (1 << 61) - 1
MANIFEST_FILENAME = "artifact_manifest.json"


def _relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _compute_payload_hash(path: Path) -> str:
    """Compute sha256 over proof payload, skipping optional 6-byte header."""
    magic = b"\xBE\xF0\xC0\xDE"
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        header = fh.read(6)
        if len(header) < 6:
            hasher.update(header)
        elif header[:4] == magic:
            remainder = header[6:]
            if remainder:
                hasher.update(remainder)
        else:
            hasher.update(header)
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_private_key(path: Path) -> bytes:
    data = path.read_text().strip()
    if data.startswith("0x"):
        data = data[2:]
    try:
        return bytes.fromhex(data)
    except ValueError as exc:
        raise ValueError("private key must be hex-encoded") from exc


def derive_r_challenges(seed: int, m: int) -> list[int]:
    base = seed.to_bytes(32, "big", signed=False)
    out: list[int] = []
    counter = 0
    while len(out) < m:
        hh = hashlib.sha256(base + counter.to_bytes(4, "big")).digest()
        val = int.from_bytes(hh, "big") % MODULUS
        if val != 0:
            out.append(val)
        counter += 1
    return out


def build_fri_cfg(steps: int, num_queries: int) -> FRIConfig:
    domain_size = 1 << (steps - 1).bit_length()
    max_rounds = max(1, domain_size.bit_length() - 1)
    return FRIConfig(
        field_modulus=MODULUS,
        domain_size=domain_size,
        max_degree=steps - 1,
        num_rounds=min(6, max_rounds),
        num_queries=num_queries,
    )


def required_samples(delta: float, epsilon: float) -> int:
    return max(1, math.ceil(math.log(1.0 / epsilon) / delta))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full geometry pipeline")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-challenges", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("out/demo_geom"))
    parser.add_argument("--trace-id", type=str, default="geom_demo")
    parser.add_argument("--challenge-seed", type=int, default=42)
    parser.add_argument(
        "--stats-out",
        type=Path,
        help="write aggregated pipeline stats to this JSON file",
    )
    parser.add_argument(
        "--no-compressed",
        action="store_true",
        help="skip compressed SNARK (Nova only)",
    )
    parser.add_argument(
        "--skip-nova",
        action="store_true",
        help="skip Nova recursion step and emit a capsule-lite",
    )
    parser.add_argument(
        "--prev-capsule-hash",
        type=str,
        help="optional previous capsule hash for chaining",
    )
    parser.add_argument(
        "--anchor-ref",
        type=str,
        help="optional external anchor reference (e.g. L1 block hash)",
    )
    parser.add_argument(
        "--anchor-rule-id",
        type=str,
        default="unspecified",
        help="identifier describing the anchor/anti-grinding rule",
    )
    parser.add_argument(
        "--row-archive-dir",
        type=Path,
        help="persist STC row chunk archive here (default: <output_dir>/row_archive)",
    )
    parser.add_argument(
        "--artifact-formats",
        choices=["json", "bin", "both"],
        default="both",
        help="emit artifacts in JSON, binary, or both",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="path to policy file whose hash is embedded in the capsule",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        required=True,
        help="identifier for the policy being enforced",
    )
    parser.add_argument(
        "--policy-version",
        type=str,
        default="unspecified",
        help="version string for the policy",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        help="path to secp256k1 private key (hex) used to sign the capsule",
    )
    parser.add_argument(
        "--da-policy-id",
        type=str,
        default="light_sampling_local_v1",
        help="identifier for the DA policy embedded in the capsule",
    )
    parser.add_argument(
        "--da-k-samples",
        type=int,
        help="override number of DA sampling chunks",
    )
    parser.add_argument(
        "--da-provider-timeout-ms",
        type=int,
        default=5000,
        help="timeout per DA provider request in milliseconds",
    )
    parser.add_argument(
        "--da-provider-retry-count",
        type=int,
        default=2,
        help="number of retry attempts for DA provider fetches",
    )
    parser.add_argument(
        "--encoding-id",
        type=str,
        default=ENCODING_ID,
        help="encoding id for binary artifacts",
    )
    args = parser.parse_args()
    formats = args.artifact_formats
    selected_encoding = args.encoding_id
    want_json = formats in {"json", "both"}
    want_bin = formats in {"bin", "both"}
    policy_path = args.policy.expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"policy file missing: {policy_path}")
    policy_digest = _compute_file_hash(policy_path)
    private_key_bytes = None
    if args.private_key:
        if PrivateKey is None:
            raise RuntimeError(
                "coincurve package required for signing; install via pip install coincurve"
            )
        private_key_path = args.private_key.expanduser().resolve()
        if not private_key_path.exists():
            raise FileNotFoundError(f"private key file missing: {private_key_path}")
        private_key_bytes = _load_private_key(private_key_path)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema": "bef_artifacts_manifest_v1",
        "version": 1,
        "encoding_id": selected_encoding,
        "path_index": {},
        "proofs": {},
    }

    def register_path(path: Path | None, encoding_id: str) -> None:
        if not path:
            return
        rel = os.path.relpath(path.resolve(), out_dir.resolve())
        manifest.setdefault("path_index", {})[rel] = {"encoding_id": encoding_id}

    params = GeomAIRParams(
        steps=args.steps,
        num_challenges=args.num_challenges,
        r_challenges=derive_r_challenges(args.challenge_seed, args.num_challenges),
        matrix=[[2, 1], [1, 1]],
    )
    init = GeomInitialState()
    t_trace = time.perf_counter()
    trace = simulate_trace(GEOM_PROGRAM, params, init)
    trace_time = time.perf_counter() - t_trace
    row_width = len(flatten_geom_row(trace.rows[0]))
    columns = column_names(params)
    schema_doc = {"columns": columns}
    schema_hash = hashlib.sha256(json.dumps(schema_doc, sort_keys=True).encode()).hexdigest()
    trace_spec = TraceSpecV1(
        spec_version="1.0",
        trace_format_id="GEOM_AIR_V1",
        record_schema_ref=f"sha256:{schema_hash}",
        encoding_id=selected_encoding,
        field_modulus_id="goldilocks_61",
    )
    trace_spec_hash = compute_trace_spec_hash(trace_spec)

    default_archive = out_dir / "row_archive"
    row_archive_dir = (args.row_archive_dir or default_archive).expanduser().resolve()
    if row_archive_dir.exists():
        shutil.rmtree(row_archive_dir)
    row_archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_FILENAME
    trace_path = out_dir / "stc_trace.json"
    proof_path_json = out_dir / "geom_proof.json"
    proof_path_bin = out_dir / "geom_proof.bin"
    nova_stats_path = out_dir / "nova_stats.json"
    capsule_path_json = out_dir / "strategy_capsule.json"
    capsule_path_bin = out_dir / "strategy_capsule.bin"
    stats_path = args.stats_out or (out_dir / "pipeline_stats.json")
    trace_abs = trace_path.resolve()
    stats_abs = nova_stats_path.resolve()

    bef_trace = geom_trace_to_bef_trace(trace, args.trace_id)
    trace_path.write_text(json.dumps(bef_trace, indent=2))
    print(f"[1/4] wrote bef_trace to {trace_path}")

    fri_cfg = build_fri_cfg(args.steps, args.num_queries)
    vc = STCVectorCommitment(chunk_len=row_width)
    profile_data: dict[str, float] = {}
    row_backend_params = {"archive_dir": row_archive_dir}
    anchors_list: list[dict[str, object]] = []
    if args.anchor_ref:
        anchors_list.append({
            "anchor_rule_id": args.anchor_rule_id,
            "anchor_ref": args.anchor_ref,
        })
    statement_holder: dict[str, object] = {}

    def _build_statement_hash(row_commitment, geom_statement):
        public_inputs = [
            {"name": "final_m11", "value": int(geom_statement.final_m11)},
            {"name": "final_m12", "value": int(geom_statement.final_m12)},
            {"name": "final_m22", "value": int(geom_statement.final_m22)},
            {"name": "final_cnt", "value": int(geom_statement.final_cnt)},
        ]
        statement = StatementV1(
            statement_version="1.0",
            trace_spec_hash=trace_spec_hash,
            policy_hash=policy_digest,
            trace_root=row_commitment.params.get("root", ""),
            public_inputs=public_inputs,
            anchors=anchors_list,
        )
        statement_hash_hex = compute_statement_hash(statement)
        statement_holder["statement"] = statement
        statement_holder["statement_hash"] = statement_hash_hex
        return bytes.fromhex(statement_hash_hex)

    proof = zk_prove_geom(
        GEOM_PROGRAM,
        params,
        init,
        fri_cfg,
        vc,
        row_backend_params=row_backend_params,
        profile=profile_data,
        statement_hash_fn=_build_statement_hash,
    )
    chunk_leaf_enabled = any(
        bool(batch.proof.chunk_leaf_proofs)
        for batch in proof.fri_proof.batches
    )
    leaf_base = int(profile_data.get("fri_leaf_bytes_baseline", 0))
    leaf_actual = int(
        profile_data.get(
            "fri_leaf_bytes_actual",
            profile_data.get("fri_leaf_siblings", 0) * 32,
        )
    )
    leaf_saved = max(leaf_base - leaf_actual, 0)
    leaf_ratio = leaf_saved / leaf_base if leaf_base else 0.0
    profile_data.setdefault("leaf_auth_bytes_baseline", leaf_base)
    profile_data.setdefault("leaf_auth_bytes_actual", leaf_actual)
    profile_data.setdefault("leaf_auth_bytes_saved", leaf_saved)
    profile_data.setdefault("leaf_auth_savings_ratio", leaf_ratio)
    if leaf_base:
        print(
            f"[stats] Leaf auth savings: {leaf_saved} bytes ({leaf_ratio:.1%})"
        )
    if (
        leaf_base
        and args.steps == 4096
        and args.num_queries == 32
        and leaf_ratio < 0.25
    ):
        raise RuntimeError(
            "leaf auth savings ratio below 25% for 4096x32 benchmark"
        )
    row_commitment = proof.row_commitment
    if row_commitment is None:
        raise RuntimeError("Geom proof is missing row commitment data")
    row_archive_rel = os.path.relpath(row_archive_dir, out_dir)
    row_commitment.params["archive_root_abs"] = str(row_archive_dir)
    row_commitment.params["archive_root"] = row_archive_rel
    chunk_handles = list(row_commitment.params.get("chunk_handles", []) or [])
    chunk_roots_hex = list(row_commitment.params.get("chunk_roots_hex", []) or [])
    row_commitment.params.pop("chunk_handles", None)
    row_commitment.params.pop("chunk_roots_hex", None)
    chunk_roots_path = row_archive_dir / "chunk_roots.json"
    chunk_roots_path.write_text(json.dumps(chunk_roots_hex, indent=2))
    chunk_roots_bin_path = row_archive_dir / "chunk_roots.bin"
    chunk_roots_bin_path.write_bytes(b"".join(bytes.fromhex(h) for h in chunk_roots_hex))
    chunk_roots_digest = _compute_file_hash(chunk_roots_bin_path)
    register_path(chunk_roots_path, "hex_json_v1")
    register_path(chunk_roots_bin_path, "raw32_v1")
    row_archive_artifact = {
        "mode": "LOCAL_FILE",
        "path": row_archive_rel,
        "abs_path": str(row_archive_dir),
        "num_chunks": len(chunk_handles),
        "chunk_tree_arity": row_commitment.params.get("chunk_tree_arity"),
        "chunk_roots_path": os.path.join(row_archive_rel, "chunk_roots.json"),
        "chunk_roots_abs": str(chunk_roots_path),
        "chunk_roots_format": "hex_json_v1",
        "chunk_roots_bin_path": os.path.join(row_archive_rel, "chunk_roots.bin"),
        "chunk_roots_bin_abs": str(chunk_roots_bin_path),
        "chunk_roots_bin_format": "raw32_v1",
        "chunk_roots_digest": chunk_roots_digest,
    }
    row_archive_artifact["chunk_handles"] = chunk_handles
    manifest["chunk_roots"] = {
        "default": "json",
        "formats": {
            "json": {
                "path": _relpath(chunk_roots_path, out_dir),
                "encoding_id": "hex_json_v1",
            },
            "bin": {
                "path": _relpath(chunk_roots_bin_path, out_dir),
                "encoding_id": "raw32_v1",
            },
        },
    }
    default_k = required_samples(0.1, 1e-6)
    da_policy = {
        "policy_id": args.da_policy_id,
        "k_samples": int(args.da_k_samples or default_k),
        "provider_timeout_ms": max(0, args.da_provider_timeout_ms),
        "provider_retry_count": max(0, args.da_provider_retry_count),
        "verification_level": "probabilistic_da_sampling",
        "provider": {
            "mode": "LOCAL_FILE",
            "archive_root": row_archive_artifact["abs_path"],
        },
    }
    chunk_len = row_commitment.row_width
    num_chunks = len(chunk_handles)
    chunk_meta = {
        "num_chunks": num_chunks,
        "chunk_len": chunk_len,
        "chunk_size_bytes": chunk_len * 8,
        "data_length_bytes": chunk_len * num_chunks * 8,
        "chunking_rule_id": "fixed_range_v1",
    }
    row_index_ref = {
        "commitment_type": "merkle_root",
        "commitment": row_commitment.params.get("root"),
        "tree_arity": row_commitment.params.get("chunk_tree_arity"),
        "proof_fetch_rule_id": "proof_attached_to_chunk_v1",
        "pointer": {
            "path": row_archive_rel,
            "provider_root": row_archive_artifact["abs_path"],
        },
    }
    hashing_meta = {
        "hash_fn_id": "sha256",
        "encoding_id": selected_encoding,
    }
    anchor_meta = {
        "anchor_rule_id": args.anchor_rule_id,
        "anchor_ref": args.anchor_ref,
    }
    manifest_geom_formats: dict[str, dict[str, str]] = {}
    proof_json = proof_to_json(proof)
    proof_size_bytes = len(proof_json.encode("utf-8"))
    if want_json:
        proof_path_json.write_text(proof_json)
        register_path(proof_path_json, "json_hex_v1")
        manifest_geom_formats["json"] = {
            "path": _relpath(proof_path_json, out_dir),
            "encoding_id": "json_hex_v1",
            "sha256_payload_hash": _compute_payload_hash(proof_path_json),
        }
        print(f"[2/4] wrote Geom proof to {proof_path_json}")
    proof_bin_bytes = proof_to_bytes(proof, encoding_id=selected_encoding)
    if want_bin:
        proof_path_bin.write_bytes(proof_bin_bytes)
        register_path(proof_path_bin, selected_encoding)
        manifest_geom_formats["bin"] = {
            "path": _relpath(proof_path_bin, out_dir),
            "encoding_id": selected_encoding,
            "sha256_payload_hash": _compute_payload_hash(proof_path_bin),
        }
        print(f"[2/4b] wrote Geom proof (bin) to {proof_path_bin}")

    statement_obj = statement_holder.get("statement")
    statement_hash_hex = statement_holder.get("statement_hash")
    if statement_obj is None or statement_hash_hex is None:
        raise RuntimeError("failed to compute StatementV1 hash")

    verify_start = time.perf_counter()
    ok, verify_stats = zk_verify_geom(
        GEOM_PROGRAM,
        params,
        init,
        STCVectorCommitment(chunk_len=row_width),
        proof_from_json(proof_json),
        statement_hash=bytes.fromhex(statement_hash_hex),
    )
    verify_time = time.perf_counter() - verify_start
    if not ok:
        raise RuntimeError("Geom proof failed verification")

    nova_stats: dict = {}
    if not args.skip_nova:
        cmd = [
            "cargo",
            "run",
            "-p",
            "nova_stc",
            "--",
            "prove",
            "--chunks",
            str(trace_abs),
            "--challenges",
            str(args.num_challenges),
            "--stats-out",
            str(stats_abs),
        ]
        if not args.no_compressed:
            cmd.append("--compressed")
        print(f"[3/4] running Nova CLI: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT / "nova_stc")
        nova_stats = json.loads(nova_stats_path.read_text())

    primary_geom_path = proof_path_json if want_json else proof_path_bin
    primary_geom_size = proof_size_bytes if want_json else len(proof_bin_bytes)
    geom_formats: dict[str, dict[str, object]] = {}
    if want_json:
        geom_formats["json"] = {
            "path": str(proof_path_json),
            "size_bytes": proof_size_bytes,
            "sha256_payload_hash": manifest_geom_formats["json"]["sha256_payload_hash"],
        }
    if want_bin:
        geom_formats["bin"] = {
            "path": str(proof_path_bin),
            "size_bytes": len(proof_bin_bytes),
            "sha256_payload_hash": manifest_geom_formats["bin"]["sha256_payload_hash"],
        }
    manifest["proofs"]["geom"] = {
        "default_format": "json" if want_json else "bin",
        "formats": manifest_geom_formats,
        "features": {"chunk_leaf_multiproof": chunk_leaf_enabled},
    }

    capsule = {
        "schema": "bef_capsule_v1",
        "vm_id": "geom_vm_v1",
        "air_id": "geom_vm_v1",
        "trace_id": args.trace_id,
        "prev_capsule_hash": args.prev_capsule_hash,
        "trace_spec": trace_spec.to_obj(),
        "trace_spec_hash": trace_spec_hash,
        "policy": {
            "policy_id": args.policy_id,
            "policy_version": args.policy_version,
            "policy_path": str(policy_path),
            "policy_hash": policy_digest,
        },
        "params": {
            "steps": args.steps,
            "num_challenges": args.num_challenges,
            "num_queries": args.num_queries,
            "row_width": row_width,
            "challenge_seed": args.challenge_seed,
            "r_challenges": params.r_challenges,
        },
        "da_policy": da_policy,
        "chunk_meta": chunk_meta,
        "row_index_ref": row_index_ref,
        "hashing": hashing_meta,
        "anchor": anchor_meta,
        "trace_commitment": nova_stats.get("nova_state") if nova_stats else None,
        "proofs": {
            "geom": {
                "path": str(primary_geom_path),
                "size_bytes": primary_geom_size,
                "row_openings": len(proof.row_openings),
                "row_backend": row_commitment.backend,
                "row_archive": row_archive_artifact,
                "formats": geom_formats,
            },
            **(
                {
                    "nova": {
                        "stats_path": str(nova_stats_path),
                        "recursive_proof_bytes": nova_stats.get("recursive_proof_bytes"),
                        "compressed": nova_stats.get("compressed"),
                    }
                }
                if nova_stats
                else {}
            ),
        },
        "row_archive": row_archive_artifact,
        "artifacts": {
            "trace": str(trace_path),
            "geom_proof": str(primary_geom_path),
            **({"nova_stats": str(nova_stats_path)} if nova_stats else {}),
            "pipeline_stats": str(stats_path),
            "row_archive": row_archive_artifact,
            "manifest": str(manifest_path),
        },
    }
    if want_bin:
        capsule["artifacts"]["strategy_capsule_bin"] = str(capsule_path_bin)
    geom_entry = capsule["proofs"]["geom"]
    if want_json:
        geom_entry["json_path"] = str(proof_path_json)
    if want_bin:
        geom_entry["bin_path"] = str(proof_path_bin)
    capsule_manifest_formats: dict[str, dict[str, str]] = {}
    if want_json:
        register_path(capsule_path_json, "json_hex_v1")
        capsule_manifest_formats["json"] = {
            "path": _relpath(capsule_path_json, out_dir),
            "encoding_id": "json_hex_v1",
        }
    if want_bin:
        register_path(capsule_path_bin, selected_encoding)
        capsule_manifest_formats["bin"] = {
            "path": _relpath(capsule_path_bin, out_dir),
            "encoding_id": selected_encoding,
        }
    manifest["capsule"] = {
        "default_format": "json" if want_json else "bin",
        "formats": capsule_manifest_formats,
    }
    capsule["statement"] = statement_obj.to_obj()
    capsule["statement_hash"] = statement_hash_hex

    capsule_for_hash = deepcopy(capsule)
    capsule_hash = compute_capsule_hash(capsule_for_hash, encoding_id=selected_encoding)
    capsule["capsule_hash"] = capsule_hash
    if private_key_bytes:
        if len(private_key_bytes) != 32:
            raise ValueError("secp256k1 private key must be 32 bytes")
        signer = PrivateKey(private_key_bytes)
        hash_bytes = bytes.fromhex(capsule_hash)
        signature = signer.sign_recoverable(hash_bytes, hasher=None)
        pubkey = signer.public_key.format(compressed=False)
        capsule["authorship"] = {
            "signer_pubkey": pubkey.hex(),
            "signature": signature.hex(),
        }
    capsule_json = json.dumps(capsule, indent=2)
    if want_json:
        capsule_path_json.write_text(capsule_json)
        print(f"[4/4] wrote capsule descriptor to {capsule_path_json}")
    if want_bin:
        capsule_path_bin.write_bytes(canonical_encode(capsule, encoding_id=selected_encoding))
        print(f"[4/4b] wrote capsule (bin) to {capsule_path_bin}")
    register_path(manifest_path, "json_manifest_v1")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[4/4c] wrote artifact manifest to {manifest_path}")

    pipeline_stats = {
        "schema": "geom_pipeline_stats_v1",
        "trace_id": args.trace_id,
        "steps": args.steps,
        "num_challenges": args.num_challenges,
        "num_queries": args.num_queries,
        "row_width": row_width,
        "challenge_seed": args.challenge_seed,
        "trace_chunks": len(bef_trace["chunks"]),
        "trace_time_sec": trace_time,
        "geom_proof": {
            "size_bytes": proof_size_bytes,
            "profile": profile_data,
            "verify_stats": verify_stats,
            "verify_time_sec": verify_time,
            "leaf_auth_bytes_baseline": leaf_base,
            "leaf_auth_bytes_actual": leaf_actual,
            "leaf_auth_bytes_saved": leaf_saved,
            "leaf_auth_savings_ratio": leaf_ratio,
        },
        "row_archive": row_archive_artifact,
        "da_policy": da_policy,
        "chunk_meta": chunk_meta,
        "row_index_ref": row_index_ref,
        **({"nova": nova_stats} if nova_stats else {}),
    }
    stats_path.write_text(json.dumps(pipeline_stats, indent=2))
    print(f"[4b] wrote pipeline stats to {stats_path}")


if __name__ == "__main__":
    main()
