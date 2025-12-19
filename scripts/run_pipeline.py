#!/usr/bin/env python3
"""Adapter-driven pipeline orchestrator for BEF capsules."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from backends import ADAPTERS
from bef_zk.adapter import TraceAdapter
from bef_zk.codec import ENCODING_ID, canonical_encode, compute_capsule_hash
from bef_zk.spec import StatementV1, compute_statement_hash
from capsule_bench.events import EventLogger, ProgressSink

try:
    from coincurve import PrivateKey
except ImportError:  # pragma: no cover
    PrivateKey = None


def _relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _compute_payload_hash(path: Path) -> str:
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


def _portable_entry(path: Path, *, rel_path: str | None = None, base: Path | None = None) -> dict[str, str]:
    entry: dict[str, str] = {"path": str(path)}
    if rel_path:
        entry["rel_path"] = rel_path
    elif base is not None:
        entry["rel_path"] = _relpath(path, base)
    return entry


def required_samples(delta: float, epsilon: float) -> int:
    return max(1, math.ceil(math.log(1.0 / epsilon) / delta))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adapter-driven BEF pipeline", conflict_handler="resolve")
    parser.add_argument("--backend", type=str, default="geom", help="trace adapter id")
    parser.add_argument("--trace-id", type=str, default="demo_trace")
    parser.add_argument("--output-dir", type=Path, default=Path("out/demo"))
    parser.add_argument("--artifact-formats", choices=["json", "bin", "both"], default="both")
    parser.add_argument("--encoding-id", type=str, default=ENCODING_ID)
    parser.add_argument("--row-archive-dir", type=Path, help="override row archive output dir")
    parser.add_argument("--stats-out", type=Path, help="pipeline stats output path")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-id", type=str, required=True)
    parser.add_argument("--policy-version", type=str, default="unspecified")
    parser.add_argument("--private-key", type=Path, help="secp256k1 private key for signing the capsule")
    parser.add_argument("--prev-capsule-hash", type=str)
    parser.add_argument("--anchor-ref", type=str)
    parser.add_argument("--anchor-rule-id", type=str, default="unspecified")
    parser.add_argument("--track-id", type=str, default="unspecified")
    parser.add_argument("--docker-image-digest", type=str)
    parser.add_argument("--events-log", type=Path, help="path to events.jsonl log")
    parser.add_argument("--da-policy-id", type=str, default="light_sampling_local_v1")
    parser.add_argument("--da-k-samples", type=int)
    parser.add_argument("--da-provider-timeout-ms", type=int, default=5000)
    parser.add_argument("--da-provider-retry-count", type=int, default=2)
    return parser


def _get_adapter_cls(name: str):
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise SystemExit(f"unknown backend '{name}'") from exc


def _prepare_manifest(output_dir: Path, encoding_id: str) -> tuple[dict[str, Any], Any]:
    manifest: dict[str, Any] = {
        "schema": "bef_artifacts_manifest_v1",
        "version": 1,
        "encoding_id": encoding_id,
        "path_index": {},
        "proofs": {},
    }

    def register_path(path: Path | None, enc_id: str) -> None:
        if not path:
            return
        rel = _relpath(path, output_dir)
        manifest.setdefault("path_index", {})[rel] = {"encoding_id": enc_id}

    return manifest, register_path



def main() -> None:
    parser = _build_parser()
    preliminary, _ = parser.parse_known_args()
    adapter_cls = _get_adapter_cls(preliminary.backend)
    adapter_cls.add_arguments(parser)
    args = parser.parse_args()
    adapter = adapter_cls(args)

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
            raise RuntimeError("coincurve package required for signing")
        pk_path = args.private_key.expanduser().resolve()
        if not pk_path.exists():
            raise FileNotFoundError(f"private key file missing: {pk_path}")
        data = pk_path.read_text().strip()
        if data.startswith("0x"):
            data = data[2:]
        private_key_bytes = bytes.fromhex(data)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_log_path = (args.events_log or (out_dir / "events.jsonl")).expanduser().resolve()
    event_logger = EventLogger(
        events_log_path,
        run_id=args.trace_id,
        trace_id=args.trace_id,
        default_source="pipeline",
    )
    adapter.set_progress_callback(ProgressSink(event_logger).callback)
    event_logger.emit(
        "run_started",
        {
            "backend": args.backend,
            "policy_id": args.policy_id,
            "track_id": args.track_id,
            "trace_id": args.trace_id,
        },
    )
    manifest, register_path = _prepare_manifest(out_dir, selected_encoding)

    default_archive = out_dir / "row_archive"
    row_archive_dir = (args.row_archive_dir or default_archive).expanduser().resolve()
    if row_archive_dir.exists():
        shutil.rmtree(row_archive_dir)
    row_archive_dir.mkdir(parents=True, exist_ok=True)

    trace_artifacts = adapter.simulate_trace(args)
    bef_trace = trace_artifacts.bef_trace
    trace_path = out_dir / "stc_trace.json"
    trace_path.write_text(json.dumps(bef_trace, indent=2))
    print(f"[1/4] wrote bef_trace to {trace_path}")
    event_logger.emit(
        "spec_locked",
        {
            "trace_spec_hash": trace_artifacts.trace_spec_hash,
            "policy_hash": policy_digest,
            "track_id": args.track_id,
        },
    )

    anchor_meta = {
        "anchor_rule_id": args.anchor_rule_id,
        "anchor_ref": args.anchor_ref,
        "track_id": args.track_id,
    }
    if args.docker_image_digest:
        anchor_meta["docker_image_digest"] = args.docker_image_digest
    trace_commitment = adapter.commit_to_trace(
        trace_artifacts,
        row_archive_dir=row_archive_dir,
    )
    row_commitment = trace_commitment.row_commitment
    row_archive_rel = _relpath(row_archive_dir, out_dir)
    row_commitment.params["archive_root_abs"] = str(row_archive_dir)
    row_commitment.params["archive_root"] = row_archive_rel

    anchors_list: list[dict[str, object]] = [dict(anchor_meta)] if anchor_meta else []

    public_inputs = adapter.extract_public_inputs(trace_artifacts)
    statement_event_hash = event_logger.emit(
        "statement_locked",
        {
            "trace_root": row_commitment.params.get("root"),
            "public_input_count": len(public_inputs),
        },
    )
    anchor_meta["event_chain_head"] = statement_event_hash
    anchors_list = [dict(anchor_meta)] if anchor_meta else []
    statement_obj = StatementV1(
        statement_version="1.0",
        trace_spec_hash=trace_artifacts.trace_spec_hash,
        policy_hash=policy_digest,
        trace_root=row_commitment.params.get("root", ""),
        public_inputs=public_inputs,
        anchors=anchors_list,
    )
    statement_hash_hex = compute_statement_hash(statement_obj)

    proof_artifacts = adapter.generate_proof(
        trace_artifacts,
        trace_commitment,
        statement_hash=bytes.fromhex(statement_hash_hex),
        encoding_id=selected_encoding,
        trace_path=trace_path,
    )

    chunk_handles = trace_commitment.chunk_handles
    chunk_roots_paths = trace_commitment.chunk_roots_paths
    register_path(chunk_roots_paths.get("json"), "hex_json_v1")
    register_path(chunk_roots_paths.get("bin"), "raw32_v1")
    row_archive_artifact = {
        **trace_commitment.row_archive_artifact,
        "mode": "LOCAL_FILE",
        "path": row_archive_rel,
        "rel_path": row_archive_rel,
        "abs_path": str(row_archive_dir),
        "chunk_roots_path": os.path.join(row_archive_rel, Path(chunk_roots_paths["json"]).name),
        "chunk_roots_abs": str(chunk_roots_paths["json"]),
        "chunk_roots_bin_path": os.path.join(row_archive_rel, Path(chunk_roots_paths["bin"]).name),
        "chunk_roots_bin_abs": str(chunk_roots_paths["bin"]),
    }
    row_archive_artifact["chunk_roots_rel_path"] = row_archive_artifact["chunk_roots_path"]
    row_archive_artifact["chunk_roots_bin_rel_path"] = row_archive_artifact["chunk_roots_bin_path"]
    row_archive_artifact["chunk_handles"] = chunk_handles

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
    register_path(events_log_path, "events_log_v1")

    proof_json_path = out_dir / "adapter_proof.json"
    proof_bin_path = out_dir / "adapter_proof.bin"
    manifest_geom_formats: dict[str, dict[str, object]] = {}
    proof_json = proof_artifacts.proof_json
    if want_json:
        proof_json_path.write_text(proof_json)
        register_path(proof_json_path, "json_hex_v1")
        manifest_geom_formats["json"] = {
            "path": _relpath(proof_json_path, out_dir),
            "encoding_id": "json_hex_v1",
            "sha256_payload_hash": _compute_payload_hash(proof_json_path),
        }
        print(f"[2/4] wrote proof JSON to {proof_json_path}")
    proof_bytes = proof_artifacts.proof_bytes
    if want_bin:
        proof_bin_path.write_bytes(proof_bytes)
        register_path(proof_bin_path, selected_encoding)
        manifest_geom_formats["bin"] = {
            "path": _relpath(proof_bin_path, out_dir),
            "encoding_id": selected_encoding,
            "sha256_payload_hash": _compute_payload_hash(proof_bin_path),
        }
        print(f"[2/4b] wrote proof BIN to {proof_bin_path}")

    primary_proof_path = proof_json_path if want_json else proof_bin_path
    primary_size = len(proof_json.encode("utf-8")) if want_json else len(proof_bytes)
    primary_proof_rel = f"proofs/primary/{primary_proof_path.name}"

    event_logger.emit(
        "proof_artifact",
        {
            "path": str(primary_proof_path),
            "size_bytes": primary_size,
        },
    )

    verify_ok, verify_stats, verify_time = adapter.verify(
        proof_json,
        bytes.fromhex(statement_hash_hex),
        trace_artifacts,
    )
    if not verify_ok:
        raise RuntimeError("adapter verification failed")

    chunk_leaf_enabled = any(
        bool(batch.proof.chunk_leaf_proofs)
        for batch in proof_artifacts.proof_obj.fri_proof.batches
    )
    manifest["chunk_roots"] = {
        "default": "json",
        "formats": {
            "json": {
                "path": _relpath(chunk_roots_paths["json"], out_dir),
                "encoding_id": "hex_json_v1",
            },
            "bin": {
                "path": _relpath(chunk_roots_paths["bin"], out_dir),
                "encoding_id": "raw32_v1",
            },
        },
    }

    capsule_path_json = out_dir / "strategy_capsule.json"
    capsule_path_bin = out_dir / "strategy_capsule.bin"
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

    # primary_proof_path/primary_size computed above for event logging
    geom_formats: dict[str, dict[str, object]] = {}
    if want_json:
        json_rel = f"proofs/primary/{proof_json_path.name}"
        geom_formats["json"] = {
            "path": str(proof_json_path),
            "rel_path": json_rel,
            "size_bytes": len(proof_json.encode("utf-8")),
            "sha256_payload_hash": manifest_geom_formats["json"]["sha256_payload_hash"],
        }
    if want_bin:
        bin_rel = f"proofs/primary/{proof_bin_path.name}"
        geom_formats["bin"] = {
            "path": str(proof_bin_path),
            "rel_path": bin_rel,
            "size_bytes": len(proof_bytes),
            "sha256_payload_hash": manifest_geom_formats["bin"]["sha256_payload_hash"],
        }
    manifest["proofs"]["primary"] = {
        "default_format": "json" if want_json else "bin",
        "formats": manifest_geom_formats,
        "features": {"chunk_leaf_multiproof": chunk_leaf_enabled},
    }

    extra_proofs = proof_artifacts.extra or {}

    capsule_artifacts: dict[str, object] = {
        "trace": _portable_entry(trace_path, rel_path=f"artifacts/{trace_path.name}"),
        "proof": _portable_entry(primary_proof_path, rel_path=primary_proof_rel),
        "row_archive": row_archive_artifact,
    }
    if events_log_path:
        capsule_artifacts["events_log"] = _portable_entry(
            events_log_path,
            rel_path=f"events/{events_log_path.name}",
        )

    capsule = {
        "schema": "bef_capsule_v1",
        "vm_id": args.backend,
        "trace_id": args.trace_id,
        "prev_capsule_hash": args.prev_capsule_hash,
        "trace_spec": trace_artifacts.trace_spec.to_obj(),
        "trace_spec_hash": trace_artifacts.trace_spec_hash,
        "policy": {
            "policy_id": args.policy_id,
            "policy_version": args.policy_version,
            "policy_path": str(policy_path),
            "policy_hash": policy_digest,
            "track_id": args.track_id,
        },
        "params": {
            "row_width": trace_artifacts.row_width,
        },
        "da_policy": da_policy,
        "chunk_meta": chunk_meta,
        "row_index_ref": row_index_ref,
        "hashing": hashing_meta,
        "anchor": anchor_meta,
        "proofs": {
            "primary": {
                "path": str(primary_proof_path),
                "rel_path": primary_proof_rel,
                "size_bytes": primary_size,
                "row_openings": len(proof_artifacts.proof_obj.row_openings),
                "row_backend": row_commitment.backend,
                "row_archive": row_archive_artifact,
                "formats": geom_formats,
            }
        },
        "row_archive": row_archive_artifact,
        "artifacts": capsule_artifacts,
    }
    if extra_proofs.get("nova"):
        nova_info = {
            "stats_path": extra_proofs["nova"].get("stats_path"),
            "recursive_proof_bytes": extra_proofs["nova"].get("recursive_proof_bytes"),
            "compressed": extra_proofs["nova"].get("compressed"),
        }
        capsule["proofs"]["nova"] = nova_info
        stats_path = extra_proofs["nova"].get("stats_path")
        if stats_path:
            path_obj = Path(stats_path)
            register_path(path_obj, "json_hex_v1")
            capsule["artifacts"]["nova_stats"] = _portable_entry(
                path_obj,
                rel_path=f"proofs/nova/{path_obj.name}",
            )

    capsule["statement"] = statement_obj.to_obj()
    capsule["statement_hash"] = statement_hash_hex

    capsule_for_hash = deepcopy(capsule)
    capsule_hash = compute_capsule_hash(capsule_for_hash, encoding_id=selected_encoding)
    capsule["capsule_hash"] = capsule_hash
    if private_key_bytes:
        if len(private_key_bytes) != 32:
            raise ValueError("secp256k1 private key must be 32 bytes")
        signer = PrivateKey(private_key_bytes)
        signature = signer.sign_recoverable(bytes.fromhex(capsule_hash), hasher=None)
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
        print(f"[4/4b] wrote capsule BIN to {capsule_path_bin}")
    event_logger.emit(
        "capsule_sealed",
        {
            "capsule_hash": capsule_hash,
        },
    )

    manifest_path = out_dir / "artifact_manifest.json"
    register_path(manifest_path, "json_manifest_v1")
    manifest["capsule"] = {
        "default_format": "json" if want_json else "bin",
        "formats": capsule_manifest_formats,
    }
    capsule.setdefault("artifacts", {})["manifest"] = _portable_entry(
        manifest_path,
        rel_path="manifests/artifact_manifest.json",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    stats_path = args.stats_out or (out_dir / "pipeline_stats.json")
    combined_profile = dict(trace_commitment.profile_data)
    combined_profile.update(proof_artifacts.profile_data)
    pipeline_stats = {
        "schema": "adapter_pipeline_stats_v1",
        "trace_id": args.trace_id,
        "row_width": trace_artifacts.row_width,
        "trace_chunks": len(bef_trace.get("chunks", [])),
        "trace_time_sec": trace_artifacts.trace_time_sec,
        "proof": {
            "size_bytes": len(proof_json.encode("utf-8")) if want_json else len(proof_bytes),
            "profile": combined_profile,
            "verify_stats": verify_stats,
            "verify_time_sec": verify_time,
            **proof_artifacts.chunk_leaf_stats,
        },
        "row_archive": row_archive_artifact,
        "da_policy": da_policy,
        "chunk_meta": chunk_meta,
        "row_index_ref": row_index_ref,
    }
    if extra_proofs.get("nova"):
        pipeline_stats["nova"] = extra_proofs["nova"]
    stats_path.write_text(json.dumps(pipeline_stats, indent=2))
    print(f"[4b] wrote pipeline stats to {stats_path}")
    event_logger.emit(
        "run_completed",
        {
            "status": "ok",
        },
    )
    event_logger.close()


if __name__ == "__main__":
    main()
