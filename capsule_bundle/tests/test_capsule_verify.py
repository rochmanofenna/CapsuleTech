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
from bef_zk.stc.merkle import build_kary_levels, root_from_levels
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


def _write_capsule(
    tmpdir: Path,
    *,
    mutate_proof: Callable[[dict], None] | None = None,
    add_archive: bool = False,
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
        "policy_id": "test_policy",
        "policy_version": "v1",
        "policy_hash": default_policy_hash,
    }
    anchor_meta = {"anchor_rule_id": "unspecified", "anchor_ref": None}
    proof_archive_dir = tmpdir / "proof_row_archive"
    if proof_archive_dir.exists():
        shutil.rmtree(proof_archive_dir)
    proof_archive_dir.mkdir(exist_ok=True)
    anchors_list = [anchor_meta] if anchor_meta.get("anchor_ref") else []
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

    artifacts = {
        "geom_proof": str(proof_path),
        "nova_stats": str(nova_stats_path),
    }
    if add_archive:
        archive_dir = tmpdir / "row_archive"
        archive_dir.mkdir(exist_ok=True)
        (archive_dir / "chunk_0.json").write_text("[]")
        artifacts["row_archive"] = {
            "mode": "LOCAL_FILE",
            "path": str(archive_dir.relative_to(tmpdir)),
        }

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
    row_params = proof.row_commitment.params if proof.row_commitment else {}
    row_root = row_params.get("root")
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
    capsule_path = tmpdir / "strategy_capsule.json"
    capsule_path.write_text(json.dumps(capsule))
    return capsule_path, nova_stats_path, nova_stats


def _attach_policy(tmpdir: Path, policy_hash: str, policy_id: str = "test_policy") -> Path:
    cap_path, _, _ = _write_capsule(tmpdir)
    capsule = json.loads(cap_path.read_text())
    capsule["policy"] = {
        "policy_id": policy_id,
        "policy_version": "v1",
        "policy_hash": policy_hash,
    }
    _refresh_statement(capsule)
    cap_path.write_text(json.dumps(capsule))
    return cap_path


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
    anchors = []
    if anchor_meta and anchor_meta.get("anchor_ref"):
        anchors.append(anchor_meta)
    statement.anchors = anchors
    capsule["statement"] = statement.to_obj()
    capsule["statement_hash"] = compute_statement_hash(statement)


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
    capsule_copy = copy.deepcopy(capsule)
    capsule_copy.pop("capsule_hash", None)
    capsule["capsule_hash"] = compute_capsule_hash(capsule_copy)
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
        proof_dict["statement"]["final_cnt"] += 1

    cap_path, _, _ = _write_capsule(cap_dir, mutate_proof=mutate)
    with pytest.raises(RuntimeError):
        verify_capsule(cap_path)


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
    archive_dir = cap_dir / "row_archive"
    shutil.rmtree(archive_dir)
    with pytest.raises(FileNotFoundError):
        verify_capsule(cap_path)


def test_verify_capsule_policy_binding(tmp_path: Path, monkeypatch) -> None:
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("policy v1")
    policy_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    capsule_path = _attach_policy(tmp_path, policy_hash)

    proof_path = tmp_path / "policy.proof"
    proof_path.write_text(json.dumps({"proof": [], "hash": policy_hash}))
    result = verify_capsule(
        capsule_path,
        policy_path=policy_file,
        policy_proof_path=proof_path,
        policy_registry_root=policy_hash,
    )
    assert result["policy_verified"] is True


def test_verify_capsule_policy_mismatch(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("policy v1")
    policy_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    capsule_path = _attach_policy(tmp_path, policy_hash)
    # tamper policy file
    policy_file.write_text("policy tampered")
    proof_path = tmp_path / "policy.proof"
    proof_path.write_text(json.dumps({"proof": [], "hash": policy_hash}))
    with pytest.raises(ValueError):
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )


def test_verify_capsule_signature_and_acl(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("policy v1")
    policy_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    capsule_path = _attach_policy(tmp_path, policy_hash, policy_id="signed_policy")
    proof_path = tmp_path / "policy.proof"
    proof_path.write_text(json.dumps({"proof": [], "hash": policy_hash}))
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
        policy_proof_path=proof_path,
        policy_registry_root=policy_hash,
        acl_path=acl_path,
    )
    assert result["authorship_verified"]
    assert result["acl_authorized"]


def test_verify_capsule_signature_invalid(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("policy v1")
    policy_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    capsule_path = _attach_policy(tmp_path, policy_hash, policy_id="signed_policy")
    proof_path = tmp_path / "policy.proof"
    proof_path.write_text(json.dumps({"proof": [], "hash": policy_hash}))
    _attach_authorship(capsule_path, "1".zfill(64))
    tampered = json.loads(capsule_path.read_text())
    tampered["authorship"]["signature"] = "00" * 65
    capsule_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        verify_capsule(
            capsule_path,
            policy_path=policy_file,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )


def test_verify_capsule_acl_rejects_unauthorized(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("policy v1")
    policy_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    capsule_path = _attach_policy(tmp_path, policy_hash, policy_id="signed_policy")
    proof_path = tmp_path / "policy.proof"
    proof_path.write_text(json.dumps({"proof": [], "hash": policy_hash}))
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
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
            acl_path=acl_path,
        )


def test_statement_hash_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "stmt_hash_mismatch"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir)
    capsule = json.loads(capsule_path.read_text())
    capsule["statement_hash"] = "00" * 32
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
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(capsule_path)
    assert "E053" in str(excinfo.value)
