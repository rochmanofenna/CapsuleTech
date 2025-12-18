from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState
from bef_zk.fri.config import FRIConfig
from scripts.geom_programs import GEOM_PROGRAM
from scripts.verify_capsule import verify_capsule
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.stc.aok_cpu import merkle_from_values
from bef_zk.stc.merkle import build_kary_levels, root_from_levels, prove_kary
from bef_zk.zk_geom.columns import column_names
from bef_zk.zk_geom.prover import zk_prove_geom
from bef_zk.zk_geom.serialization import proof_to_json
from bef_zk.codec import compute_capsule_hash
from bef_zk.spec import (
    TraceSpecV1,
    StatementV1,
    compute_trace_spec_hash,
    compute_statement_hash,
)


def _default_params(steps: int = 8) -> GeomAIRParams:
    return GeomAIRParams(
        steps=steps,
        num_challenges=2,
        r_challenges=[1234567, 89101112],
        matrix=[[2, 1], [1, 1]],
    )


def _fri_cfg(steps: int) -> FRIConfig:
    domain_size = 1 << (steps - 1).bit_length()
    max_rounds = max(1, domain_size.bit_length() - 1)
    return FRIConfig(
        field_modulus=(1 << 61) - 1,
        domain_size=domain_size,
        max_degree=steps - 1,
        num_rounds=min(4, max_rounds),
        num_queries=4,
    )


def _row_width(params: GeomAIRParams) -> int:
    return len(column_names(params))


def _payload_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_event_chain(path: Path, trace_id: str, events: list[tuple[str, dict]]) -> str:
    prev_hash = "0" * 64
    statement_event_hash: str | None = None
    with path.open("w", encoding="utf-8") as fh:
        for seq, (event_type, data) in enumerate(events, start=1):
            base_event = {
                "schema": "bef_capsule_stream_v1",
                "v": 1,
                "trace_id": trace_id,
                "seq": seq,
                "ts_ms": seq,
                "type": event_type,
                "data": data,
            }
            serialized = json.dumps(base_event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            event_hash = hashlib.sha256(bytes.fromhex(prev_hash) + serialized).hexdigest()
            payload = dict(base_event)
            payload["prev_event_hash"] = prev_hash
            payload["event_hash"] = event_hash
            fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            if event_type == "statement_locked":
                statement_event_hash = event_hash
            prev_hash = event_hash
    if not statement_event_hash:
        raise RuntimeError("statement_locked event missing from test log")
    return statement_event_hash


def _update_capsule_hash(capsule: dict) -> None:
    capsule_copy = copy.deepcopy(capsule)
    capsule_copy.pop("capsule_hash", None)
    capsule["capsule_hash"] = compute_capsule_hash(capsule_copy)


def _sync_capsule_after_proof_change(capsule_path: Path, proof_path: Path) -> None:
    capsule = json.loads(capsule_path.read_text())
    geom_entry = (capsule.get("proofs") or {}).get("geom") or {}
    json_fmt = (geom_entry.get("formats") or {}).get("json")
    if json_fmt is not None:
        json_fmt["sha256_payload_hash"] = _payload_hash(proof_path)
    capsule.setdefault("proofs", {})["geom"] = geom_entry
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))


def _write_policy_proof(proof_path: Path, policy_hash: str) -> None:
    leaf = bytes.fromhex(policy_hash)
    levels = build_kary_levels([leaf], 2)
    siblings = prove_kary(levels, 0, 2)
    proof = {
        "leaf_hash": policy_hash,
        "leaf_index": 0,
        "total_leaves": 1,
        "arity": 2,
        "siblings_by_level": [[s.hex() for s in level] for level in siblings],
    }
    proof_path.write_text(json.dumps(proof))


def _write_policy_file(path: Path, *, track_id: str = "baseline_no_accel") -> str:
    policy = {
        "schema": "bef_benchmark_policy_v1",
        "policy_id": "baseline_policy_v1",
        "policy_version": "1.0",
        "tracks": [
            {
                "track_id": track_id,
                "rules": {
                    "forbid_gpu": True,
                    "require_deterministic_build": True,
                    "required_public_outputs": ["final_cnt"],
                },
            }
        ],
    }
    path.write_text(json.dumps(policy))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_capsule(
    tmpdir: Path,
    *,
    mutate_proof: Callable[[dict], None] | None = None,
    add_archive: bool = False,
    policy_hash_override: str | None = None,
    policy_id: str = "test_policy",
    track_id: str = "test_track",
    anchor_ref: str | None = None,
    docker_image_digest: str | None = None,
) -> tuple[Path, Path, dict]:
    params = _default_params()
    init = GeomInitialState()
    fri_cfg = _fri_cfg(params.steps)
    vc = STCVectorCommitment(chunk_len=256)
    columns = column_names(params)
    schema_doc = {"columns": columns}
    schema_hash = hashlib.sha256(json.dumps(schema_doc, sort_keys=True).encode()).hexdigest()
    trace_spec = TraceSpecV1(
        spec_version="1.0",
        trace_format_id="GEOM_AIR_V1",
        record_schema_ref=f"sha256:{schema_hash}",
        encoding_id="dag_cbor_canonical_v1",
        field_modulus_id="goldilocks_61",
    )
    trace_spec_hash = compute_trace_spec_hash(trace_spec)
    default_policy_hash = hashlib.sha256(b"default_policy").hexdigest()
    policy_info = {
        "policy_id": policy_id,
        "policy_version": "v1",
        "policy_hash": policy_hash_override or default_policy_hash,
        "track_id": track_id,
    }
    anchor_meta = {
        "anchor_rule_id": "unspecified",
        "anchor_ref": anchor_ref,
        "track_id": track_id,
    }
    events_path = tmpdir / "events.jsonl"
    statement_event_hash = _write_event_chain(
        events_path,
        trace_id="test_capsule",
        events=[
            ("run_started", {"backend": "geom", "track_id": track_id}),
            ("spec_locked", {"trace_id": "test_capsule"}),
            ("statement_locked", {"trace_root": "deadbeef"}),
            ("capsule_sealed", {"status": "ok"}),
        ],
    )
    anchor_meta["event_chain_head"] = statement_event_hash
    if docker_image_digest:
        anchor_meta["docker_image_digest"] = docker_image_digest
    proof_archive_dir = tmpdir / "proof_row_archive"
    if proof_archive_dir.exists():
        shutil.rmtree(proof_archive_dir)
    proof_archive_dir.mkdir(exist_ok=True)
    anchors_list = [anchor_meta] if anchor_meta else []
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
            policy_hash=policy_info["policy_hash"],
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
        row_backend_params={"archive_dir": proof_archive_dir},
        statement_hash_fn=_build_statement_hash,
    )
    proof_dict = json.loads(proof_to_json(proof))
    if mutate_proof is not None:
        mutate_proof(proof_dict)
    proof_path = tmpdir / "geom_proof.json"
    proof_path.write_text(json.dumps(proof_dict))

    state = {
        "n": "0x0",
        "root": "0x1",
        "s": ["0x2", "0x3"],
        "pow": ["0x4", "0x5"],
    }
    nova_stats = {"nova_state": state}
    nova_stats_path = tmpdir / "nova_stats.json"
    nova_stats_path.write_text(json.dumps(nova_stats))

    row_commitment = proof.row_commitment
    if row_commitment is None:
        raise RuntimeError("expected row commitment")
    row_commitment.params["archive_root_abs"] = str(proof_archive_dir)
    row_commitment.params["archive_root"] = str(proof_archive_dir.relative_to(tmpdir))
    row_params = row_commitment.params
    artifacts = {
        "geom_proof": str(proof_path),
        "nova_stats": str(nova_stats_path),
        "events_log": str(events_path),
    }
    row_archive_artifact = None
    if add_archive:
        chunk_roots_hex = list(row_params.get("chunk_roots_hex", []))
        proof_archive_dir.mkdir(exist_ok=True)
        chunk_roots_path = proof_archive_dir / "chunk_roots.json"
        chunk_roots_path.write_text(json.dumps(chunk_roots_hex))
        row_archive_artifact = {
            "mode": "LOCAL_FILE",
            "path": str(proof_archive_dir.relative_to(tmpdir)),
            "abs_path": str(proof_archive_dir),
            "chunk_roots_path": str(chunk_roots_path),
            "chunk_roots_format": "hex_json_v1",
            "chunk_tree_arity": row_params.get("chunk_tree_arity", 2),
        }
        artifacts["row_archive"] = row_archive_artifact

    capsule = {
        "schema": "bef_capsule_v1",
        "vm_id": "geom_vm_v1",
        "air_id": "geom_vm_v1",
        "trace_id": "test_capsule",
        "params": {
            "steps": params.steps,
            "num_challenges": params.num_challenges,
            "num_queries": 4,
            "row_width": _row_width(params),
            "challenge_seed": 0,
            "r_challenges": params.r_challenges,
        },
        "trace_commitment": state,
        "artifacts": artifacts,
        "policy": policy_info,
        "trace_spec": trace_spec.to_obj(),
        "trace_spec_hash": trace_spec_hash,
        "anchor": anchor_meta,
    }
    statement = statement_holder.get("statement")
    statement_hash_hex = statement_holder.get("statement_hash")
    if statement is not None and statement_hash_hex is not None:
        capsule["statement"] = statement.to_obj()
        capsule["statement_hash"] = statement_hash_hex
    row_root = row_params.get("root") if row_params else None
    if row_root:
        capsule.setdefault("row_index_ref", {
            "commitment_type": "merkle_root",
            "commitment": row_root,
            "tree_arity": row_params.get("chunk_tree_arity", 2),
            "proof_fetch_rule_id": "test",
        })
        capsule.setdefault("chunk_meta", {
            "num_chunks": row_params.get("num_chunks"),
            "chunk_len": row_params.get("chunk_len"),
        })
    capsule["proofs"] = {
        "geom": {
            "path": str(proof_path),
            "row_openings": len(proof_dict.get("row_openings", [])),
            "row_backend": (proof_dict.get("row_commitment") or {}).get("backend", "geom_stc_fri"),
            "row_archive": row_archive_artifact,
            "formats": {
                "json": {
                    "path": str(proof_path),
                    "encoding_id": "json_hex_v1",
                    "sha256_payload_hash": _payload_hash(proof_path),
                }
            },
        }
    }
    capsule["row_archive"] = row_archive_artifact
    capsule.setdefault("hashing", {"hash_fn_id": "sha256", "encoding_id": "dag_cbor_canonical_v1"})
    _update_capsule_hash(capsule)
    capsule_path = tmpdir / "strategy_capsule.json"
    capsule_path.write_text(json.dumps(capsule))
    return capsule_path, nova_stats_path, nova_stats


def _attach_policy(
    tmpdir: Path,
    policy_hash: str,
    policy_id: str = "test_policy",
    track_id: str = "test_track",
    anchor_ref: str | None = None,
    docker_image_digest: str | None = None,
) -> Path:
    cap_path, _, _ = _write_capsule(
        tmpdir,
        policy_hash_override=policy_hash,
        policy_id=policy_id,
        track_id=track_id,
        anchor_ref=anchor_ref,
        docker_image_digest=docker_image_digest,
    )
    return cap_path


def _make_manifest_root(tmp_path: Path, *, gpu_detected: bool = False) -> tuple[Path, str]:
    root = tmp_path / "manifests"
    root.mkdir(exist_ok=True)
    hardware = {
        "schema": "bef_hardware_manifest_v1",
        "gpu": {
            "detected": gpu_detected,
            "devices": [] if not gpu_detected else [{"model": "Test GPU"}],
        },
    }
    os_manifest = {"schema": "bef_os_fingerprint_v1", "platform": "test-os"}
    toolchain = {"schema": "bef_toolchain_manifest_v1", "python": sys.version}
    manifest_index = {
        "schema": "bef_manifest_index_v1",
        "entries": [],
    }
    files = {
        "hardware_manifest": hardware,
        "os_fingerprint": os_manifest,
        "toolchain_manifest": toolchain,
        "manifest_index": manifest_index,
    }
    hashes: dict[str, str] = {}
    for name, payload in files.items():
        path = root / f"{name}.json"
        path.write_text(json.dumps(payload))
        hashes[name] = f"sha256:{_payload_hash(path)}"
    anchor_payload = json.dumps(
        {
            "schema": "capsule_bench_manifest_anchor_v1",
            "hashes": hashes,
        },
        sort_keys=True,
    ).encode()
    anchor = f"capsulebench_manifest_v1:{hashlib.sha256(anchor_payload).hexdigest()}"
    return root, anchor


def _refresh_statement(capsule: dict) -> None:
    statement_obj = capsule.get("statement")
    trace_spec_hash = capsule.get("trace_spec_hash")
    if not statement_obj or not trace_spec_hash:
        return
    statement = StatementV1.from_obj(statement_obj)
    policy_info = capsule.get("policy") or {}
    policy_hash = policy_info.get("policy_hash")
    if policy_hash:
        statement.policy_hash = policy_hash
    trace_root = (capsule.get("row_index_ref") or {}).get("commitment")
    if trace_root:
        statement.trace_root = trace_root
    anchor_meta = capsule.get("anchor")
    anchors = [anchor_meta] if anchor_meta else []
    statement.anchors = anchors
    capsule["statement"] = statement.to_obj()
    capsule["statement_hash"] = compute_statement_hash(statement)
    _update_capsule_hash(capsule)


def _attach_da_capsule(tmpdir: Path, sample_k: int = 2) -> tuple[Path, Path]:
    cap_path, _, _ = _write_capsule(tmpdir)
    capsule = json.loads(cap_path.read_text())
    proof_path = tmpdir / "geom_proof.json"
    proof = json.loads(proof_path.read_text())
    rc_params = (proof.get("row_commitment") or {}).get("params") or {}
    archive_src = Path(rc_params.get("archive_root") or "")
    if not archive_src.exists():
        raise FileNotFoundError("row commitment archive missing for test fixture")
    archive_dir = tmpdir / "row_archive_da"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    shutil.copytree(archive_src, archive_dir)
    chunk_handles = [str(h) for h in rc_params.get("chunk_handles", [])]
    chunk_roots_hex = list(rc_params.get("chunk_roots_hex", []))
    chunk_roots = [bytes.fromhex(h) for h in chunk_roots_hex]
    chunk_len = int(rc_params.get("chunk_len") or 0)
    chunk_tree_arity = int(rc_params.get("chunk_tree_arity") or 2)
    root = root_from_levels(build_kary_levels(chunk_roots, chunk_tree_arity))
    rel_path = archive_dir.relative_to(tmpdir)
    chunk_roots_path = archive_dir / "chunk_roots.json"
    chunk_roots_path.write_text(json.dumps(chunk_roots_hex, indent=2))
    chunk_roots_bin_path = archive_dir / "chunk_roots.bin"
    chunk_roots_bin_path.write_bytes(b"".join(bytes.fromhex(h) for h in chunk_roots_hex))
    chunk_roots_digest = hashlib.sha256(chunk_roots_bin_path.read_bytes()).hexdigest()
    row_archive_info = {
        "mode": "LOCAL_FILE",
        "path": rel_path.as_posix(),
        "abs_path": str(archive_dir),
        "chunk_handles": chunk_handles,
        "chunk_roots_hex": chunk_roots_hex,
        "chunk_tree_arity": chunk_tree_arity,
        "chunk_roots_digest": chunk_roots_digest,
        "chunk_roots_bin_path": str(chunk_roots_bin_path.relative_to(tmpdir)),
        "chunk_roots_bin_abs": str(chunk_roots_bin_path),
        "chunk_roots_path": str(chunk_roots_path.relative_to(tmpdir)),
        "chunk_roots_abs": str(chunk_roots_path),
    }
    artifacts = capsule.setdefault("artifacts", {})
    artifacts["row_archive"] = {
        "mode": "LOCAL_FILE",
        "path": rel_path.as_posix(),
    }
    capsule["row_archive"] = row_archive_info
    capsule["row_index_ref"] = {
        "commitment_type": "merkle_root",
        "commitment": root.hex(),
        "tree_arity": chunk_tree_arity,
        "proof_fetch_rule_id": "local_test",
        "pointer": {"path": rel_path.as_posix()},
    }
    capsule["chunk_meta"] = {
        "num_chunks": len(chunk_handles),
        "chunk_len": chunk_len,
    }
    effective_k = sample_k if sample_k >= 0 else len(chunk_handles)
    capsule["da_policy"] = {
        "policy_id": "da_local_test",
        "k_samples": effective_k,
        "provider_timeout_ms": 1000,
        "provider_retry_count": 1,
        "provider": {
            "mode": "LOCAL_FILE",
            "archive_root": str(archive_dir),
        },
    }
    _refresh_statement(capsule)
    cap_path.write_text(json.dumps(capsule))
    return cap_path, archive_dir


def _attach_authorship(cap_path: Path, private_key_hex: str) -> str:
    coincurve = pytest.importorskip("coincurve")
    PrivateKey = coincurve.PrivateKey
    capsule = json.loads(cap_path.read_text())
    capsule_copy = copy.deepcopy(capsule)
    capsule_copy.pop("capsule_hash", None)
    capsule_copy.pop("authorship", None)
    capsule_hash = compute_capsule_hash(capsule_copy)
    capsule["capsule_hash"] = capsule_hash
    priv = PrivateKey(bytes.fromhex(private_key_hex))
    signature = priv.sign_recoverable(bytes.fromhex(capsule_hash), hasher=None)
    pubkey = priv.public_key.format(compressed=False).hex()
    capsule["authorship"] = {
        "signer_pubkey": pubkey,
        "signature": signature.hex(),
    }
    cap_path.write_text(json.dumps(capsule))
    return pubkey


def test_verify_capsule_succeeds(tmp_path: Path) -> None:
    cap_dir = tmp_path / "valid"
    cap_dir.mkdir()
    cap_path, _, _ = _write_capsule(cap_dir)
    result = verify_capsule(cap_path)
    assert result["trace_commitment"]["root"] == "0x1"


def test_verify_capsule_detects_nova_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "nova_mismatch"
    cap_dir.mkdir()
    cap_path, stats_path, stats = _write_capsule(cap_dir)
    bad_state = dict(stats["nova_state"])
    bad_state["root"] = "0xdead"
    stats_path.write_text(json.dumps({"nova_state": bad_state}))
    with pytest.raises(RuntimeError):
        verify_capsule(cap_path)


def test_verify_capsule_detects_corrupted_geom_proof(tmp_path: Path) -> None:
    cap_dir = tmp_path / "geom_corrupt"
    cap_dir.mkdir()

    def mutate(proof_dict: dict) -> None:
        proof_dict["row_openings"][0]["row_values"][0] += 1

    cap_path, _, _ = _write_capsule(cap_dir, mutate_proof=mutate)
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(cap_path)
    assert "E054" in str(excinfo.value)


def test_verify_capsule_with_row_archive(tmp_path: Path) -> None:
    cap_dir = tmp_path / "with_archive"
    cap_dir.mkdir()
    cap_path, _, _ = _write_capsule(cap_dir, add_archive=True)
    result = verify_capsule(cap_path)
    assert result["row_archive"] is not None


def test_verify_capsule_missing_row_archive(tmp_path: Path) -> None:
    cap_dir = tmp_path / "missing_archive"
    cap_dir.mkdir()
    cap_path, _, _ = _write_capsule(cap_dir, add_archive=True)
    archive_dir = cap_dir / "proof_row_archive"
    shutil.rmtree(archive_dir)
    with pytest.raises(FileNotFoundError):
        verify_capsule(cap_path)


def test_verify_capsule_policy_binding(tmp_path: Path, monkeypatch) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path)
    capsule_path = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
    )
    result = verify_capsule(
        capsule_path,
        policy_path=policy_file,
        manifest_root=manifest_root,
    )
    assert result["policy_track"] == "baseline_no_accel"


def test_verify_capsule_policy_mismatch(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path)
    capsule_path = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
    )
    policy_file.write_text("tampered")
    with pytest.raises(ValueError):
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            manifest_root=manifest_root,
        )


def test_policy_forbid_gpu_violation(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path, gpu_detected=True)
    capsule_path = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
    )
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            manifest_root=manifest_root,
        )
    assert "E101" in str(excinfo.value)


def test_policy_required_output_violation(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path)
    capsule_path, _, _ = _write_capsule(
        tmp_path,
        policy_hash_override=policy_hash,
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
        mutate_proof=None,
    )
    capsule = json.loads(capsule_path.read_text())
    statement = capsule["statement"]
    statement["public_inputs"] = [entry for entry in statement["public_inputs"] if entry["name"] != "final_cnt"]
    capsule["statement"] = statement
    capsule["statement_hash"] = compute_statement_hash(StatementV1.from_obj(statement))
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            manifest_root=manifest_root,
        )
    assert "E102" in str(excinfo.value)


def test_verify_capsule_signature_and_acl(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path)
    capsule_path = _attach_policy(
        tmp_path,
        policy_hash,
        policy_id="signed_policy",
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
    )
    signer_pub = _attach_authorship(capsule_path, "1".zfill(64))
    acl_path = tmp_path / "acl.json"
    acl = {
        "schema": "bef_acl_v1",
        "authorizations": {
            "signed_policy": [
                {"pubkey": signer_pub, "status": "active", "description": "test"}
            ]
        },
    }
    acl_path.write_text(json.dumps(acl))
    result = verify_capsule(
        capsule_path,
        policy_path=policy_file,
        manifest_root=manifest_root,
        acl_path=acl_path,
    )
    assert result["authorship_verified"]
    assert result["acl_authorized"]


def test_verify_capsule_signature_invalid(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path)
    capsule_path = _attach_policy(
        tmp_path,
        policy_hash,
        policy_id="signed_policy",
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
    )
    _attach_authorship(capsule_path, "1".zfill(64))
    tampered = json.loads(capsule_path.read_text())
    tampered["authorship"]["signature"] = "00" * 65
    _update_capsule_hash(tampered)
    capsule_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            manifest_root=manifest_root,
        )


def test_verify_capsule_acl_rejects_unauthorized(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path)
    capsule_path = _attach_policy(
        tmp_path,
        policy_hash,
        policy_id="signed_policy",
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
    )
    signer_pub = _attach_authorship(capsule_path, "1".zfill(64))
    acl_path = tmp_path / "acl.json"
    acl = {
        "schema": "bef_acl_v1",
        "authorizations": {
            "signed_policy": [
                {"pubkey": "deadbeef", "status": "active"}
            ]
        },
    }
    acl_path.write_text(json.dumps(acl))
    with pytest.raises(ValueError):
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            manifest_root=manifest_root,
            acl_path=acl_path,
        )


def test_event_chain_mismatch(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    manifest_root, anchor_ref = _make_manifest_root(tmp_path)
    capsule_path = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        anchor_ref=anchor_ref,
        docker_image_digest="sha256:deadbeef",
    )
    capsule = json.loads(capsule_path.read_text())
    events_path = Path(capsule["artifacts"]["events_log"])
    events_path.write_text("tampered\n")
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            manifest_root=manifest_root,
        )
    assert "E201" in str(excinfo.value)


def test_statement_hash_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "stmt_hash_mismatch"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir)
    capsule = json.loads(capsule_path.read_text())
    capsule["statement_hash"] = "00" * 32
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(capsule_path)
    assert "E053" in str(excinfo.value)


def test_statement_trace_root_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "stmt_root_mismatch"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir)
    capsule = json.loads(capsule_path.read_text())
    capsule["statement"]["trace_root"] = "deadbeef"
    capsule["statement_hash"] = compute_statement_hash(StatementV1.from_obj(capsule["statement"]))
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(capsule_path)
    assert "E053" in str(excinfo.value)


def test_verify_capsule_da_audit_passes(tmp_path: Path) -> None:
    cap_dir = tmp_path / "da_ok"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir)
    result = verify_capsule(capsule_path)
    assert result["da_audit_verified"]


def test_verify_capsule_da_audit_detects_corruption(tmp_path: Path) -> None:
    cap_dir = tmp_path / "da_corrupt"
    cap_dir.mkdir()
    capsule_path, archive_dir = _attach_da_capsule(cap_dir, sample_k=-1)
    chunk0 = archive_dir / "chunk_0.json"
    values = json.loads(chunk0.read_text())
    values[0] += 1
    chunk0.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        verify_capsule(capsule_path)


def test_verify_capsule_detects_row_commitment_root_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "row_root_mismatch"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir)
    proof_path = cap_dir / "geom_proof.json"
    proof = json.loads(proof_path.read_text())
    proof["row_commitment"]["params"]["root"] = "00" * 32
    proof_path.write_text(json.dumps(proof))
    _sync_capsule_after_proof_change(capsule_path, proof_path)
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(capsule_path)
    assert "E053" in str(excinfo.value)


def test_verify_capsule_detects_chunk_len_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "chunk_len_mismatch"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir)
    proof_path = cap_dir / "geom_proof.json"
    proof = json.loads(proof_path.read_text())
    params = proof["row_commitment"]["params"]
    params["chunk_len"] = int(params.get("chunk_len", 0)) + 1
    proof_path.write_text(json.dumps(proof))
    _sync_capsule_after_proof_change(capsule_path, proof_path)
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(capsule_path)
    assert "E053" in str(excinfo.value)
