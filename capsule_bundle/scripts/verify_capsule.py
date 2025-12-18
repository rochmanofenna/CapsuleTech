#!/usr/bin/env python3
"""Verify a strategy capsule end-to-end with stable error codes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:  # Optional dependency for signature checks
    from coincurve import PublicKey
except ImportError:  # pragma: no cover - coincurve may be missing in CI
    PublicKey = None

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState
from bef_zk.codec import (
    ENCODING_ID,
    canonical_decode,
    compute_capsule_hash,
    derive_capsule_seed,
)
from bef_zk.da import AvailabilityError, LocalFileSystemProvider, PolicyAwareDAClient
from bef_zk.stc.aok_cpu import merkle_from_values
from bef_zk.stc.merkle import build_kary_levels, root_from_levels, verify_kary
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.zk_geom.serialization import proof_from_bytes, proof_from_json
from bef_zk.zk_geom.verifier import zk_verify_geom
from bef_zk.spec import (
    TraceSpecV1,
    StatementV1,
    compute_trace_spec_hash,
    compute_statement_hash,
)
from bef_zk.verifier_errors import *
from scripts.artifact_manifest import encoding_for_path, load_manifest
from scripts.geom_programs import GEOM_PROGRAM

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _resolve(base: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def _load_capsule(path: Path, encoding_id: str | None = None) -> tuple[dict | None, str]:
    try:
        raw = path.read_bytes()
        enc = encoding_id
        if enc is None and path.suffix == ".bin":
            enc = "dag_cbor_canonical_v1"
        data = canonical_decode(raw) if enc == "dag_cbor_canonical_v1" else json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return None, E002_PARSE_FAILED
        return data, OK
    except (json.JSONDecodeError, FileNotFoundError, IsADirectoryError):
        return None, E002_PARSE_FAILED


def _canonical_capsule_hash(capsule: dict) -> str:
    hashing_meta = capsule.get("hashing") or {}
    encoding_id = hashing_meta.get("encoding_id") or ENCODING_ID
    capsule_copy = copy.deepcopy(capsule)
    capsule_copy.pop("capsule_hash", None)
    return compute_capsule_hash(capsule_copy, encoding_id=encoding_id)


def _load_chunk_roots(info: dict | None, base: Path) -> tuple[list[bytes] | None, str]:
    if not info:
        return None, E060_ROW_INDEX_COMMITMENT_MISSING
    inline = info.get("chunk_roots_hex")
    if inline:
        try:
            return [bytes.fromhex(h) for h in inline], OK
        except ValueError:
            return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
    bin_path = info.get("chunk_roots_bin_abs") or info.get("chunk_roots_bin_path")
    if bin_path:
        try:
            resolved = _resolve(base, bin_path)
            data = resolved.read_bytes()
            if len(data) % 32 != 0:
                return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
            return [data[i : i + 32] for i in range(0, len(data), 32)], OK
        except FileNotFoundError:
            return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
    json_path = info.get("chunk_roots_abs") or info.get("chunk_roots_path")
    if json_path:
        try:
            resolved = _resolve(base, json_path)
            return [bytes.fromhex(h) for h in json.loads(resolved.read_text())], OK
        except (FileNotFoundError, json.JSONDecodeError):
            return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
    return None, E060_ROW_INDEX_COMMITMENT_MISSING


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


def _load_acl(path: Path | None) -> tuple[dict[str, list[dict[str, str]]] | None, str]:
    if not path:
        return None, OK
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None, E002_PARSE_FAILED
    if data.get("schema") and data.get("schema") != "bef_acl_v1":
        return None, E002_PARSE_FAILED
    return data.get("authorizations", {}), OK


def _verify_authorship(capsule: dict, capsule_hash: str, require: bool) -> tuple[str, str | None, bool]:
    authorship = capsule.get("authorship")
    if not authorship:
        if require:
            return E020_SIGNATURE_MISSING, None, False
        return OK, None, False
    if PublicKey is None:
        return E021_SIGNATURE_INVALID, None, False
    signature_hex = authorship.get("signature")
    pubkey_hex = authorship.get("signer_pubkey")
    if not signature_hex or not pubkey_hex:
        return E021_SIGNATURE_INVALID, None, False
    try:
        signature = bytes.fromhex(signature_hex)
        claimed = bytes.fromhex(pubkey_hex)
    except ValueError:
        return E021_SIGNATURE_INVALID, None, False
    if len(signature) != 65 or len(claimed) not in (33, 65):
        return E021_SIGNATURE_INVALID, None, False
    message = bytes.fromhex(capsule_hash)
    try:
        recovered = PublicKey.from_signature_and_message(signature, message, hasher=None)
    except Exception:  # pragma: no cover - coincurve raises ValueError
        return E021_SIGNATURE_INVALID, None, False
    recovered_bytes = recovered.format(compressed=False)
    if recovered_bytes != PublicKey(claimed).format(compressed=False):
        return E021_SIGNATURE_INVALID, None, False
    claimed_pubkey = PublicKey(claimed)
    if not claimed_pubkey.verify(signature[:-1], message, hasher=None):
        return E021_SIGNATURE_INVALID, None, False
    return OK, recovered_bytes.hex(), True


def _verify_acl(policy_id: str | None, signer_hex: str | None, acl: dict[str, list[dict[str, str]]] | None) -> tuple[str, bool]:
    if not acl:
        return OK, False
    if not signer_hex:
        return E021_SIGNATURE_INVALID, False
    if not policy_id:
        return E030_POLICY_ID_MISSING, False
    allowed = acl.get(policy_id)
    if not allowed:
        return E022_SIGNER_NOT_AUTHORIZED, False
    key = signer_hex.lower()
    for entry in allowed:
        entry_key = (entry.get("pubkey") or "").lower()
        status = (entry.get("status") or "active").lower()
        if entry_key == key and status == "active":
            return OK, True
    return E022_SIGNER_NOT_AUTHORIZED, False


def _derive_audit_seed(capsule: dict) -> int:
    anchor = (capsule.get("anchor") or {}).get("anchor_ref")
    policy = capsule.get("policy") or {}
    capsule_hash = capsule.get("capsule_hash") or ""
    return derive_capsule_seed(
        capsule_hash,
        anchor_ref=anchor,
        policy_id=policy.get("policy_id"),
        policy_version=policy.get("policy_version"),
    )


def _select_audit_indices(total: int, count: int, seed: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    selected: list[int] = []
    seen: set[int] = set()
    counter = 0
    base = seed.to_bytes(32, "big", signed=False)
    while len(selected) < count:
        digest = hashlib.sha256(base + counter.to_bytes(4, "big", signed=False)).digest()
        counter += 1
        for pos in range(0, len(digest), 4):
            if len(selected) >= count:
                break
            idx = int.from_bytes(digest[pos : pos + 4], "big") % total
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
        if counter > 1_000_000:
            raise RuntimeError("DA sampling seed generation failed")
    return selected


def _run_da_audit(
    capsule: dict,
    chunk_roots: list[bytes],
    chunk_len: int,
    row_root: bytes,
    sample_count: int,
    provider,
) -> tuple[str, bool]:
    if sample_count <= 0:
        return OK, False
    if chunk_len <= 0:
        return E066_CHUNK_LENGTH_INVALID, False
    indices = _select_audit_indices(len(chunk_roots), sample_count, _derive_audit_seed(capsule))
    if not indices:
        return OK, False
    try:
        fetched = provider.fetch_batch(indices)
    except AvailabilityError:
        return E074_AVAILABILITY_FAILED, False
    for idx in indices:
        chunk = fetched.get(idx)
        if chunk is None:
            return E074_AVAILABILITY_FAILED, False
        offset = idx * chunk_len
        derived_root = merkle_from_values(chunk.values, offset)
        expected_root = chunk_roots[idx]
        if derived_root != expected_root:
            return E065_CHUNK_ROOT_MISMATCH, False
        proof = chunk.proof
        if not verify_kary(
            row_root,
            derived_root,
            idx,
            proof.siblings,
            proof.arity,
            proof.tree_size,
        ):
            return E064_MERKLE_PROOF_INVALID, False
    return OK, True


def _verify_row_commitment_binding(
    proof_row_commitment,
    chunk_meta: dict | None,
    row_index_ref: dict | None,
) -> str:
    if proof_row_commitment is None or not chunk_meta or not row_index_ref:
        return OK
    params = proof_row_commitment.params or {}
    proof_root = params.get("root")
    capsule_root = row_index_ref.get("commitment")
    if proof_root and capsule_root:
        if str(proof_root).lower() != str(capsule_root).lower():
            return E053_PROOF_STATEMENT_MISMATCH
    chunk_len_capsule = chunk_meta.get("chunk_len")
    chunk_len_proof = params.get("chunk_len")
    if chunk_len_capsule is not None and chunk_len_proof is not None:
        if int(chunk_len_capsule) != int(chunk_len_proof):
            return E053_PROOF_STATEMENT_MISMATCH
    arity_capsule = row_index_ref.get("tree_arity")
    arity_proof = params.get("chunk_tree_arity")
    if arity_capsule is not None and arity_proof is not None:
        if int(arity_capsule) != int(arity_proof):
            return E053_PROOF_STATEMENT_MISMATCH
    return OK


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def _verify_capsule_core(
    capsule_path: Path,
    *,
    policy_path: Path | None = None,
    policy_proof_path: Path | None = None,
    policy_registry_root: str | None = None,
    acl_path: Path | None = None,
) -> tuple[str, dict | None]:
    capsule_path = capsule_path.resolve()
    base = capsule_path.parent
    manifest = load_manifest(base)
    encoding = encoding_for_path(manifest, base, capsule_path) if manifest else None

    capsule, err = _load_capsule(capsule_path, encoding)
    if err != OK:
        return err, None
    if capsule.get("schema") != "bef_capsule_v1":
        return E003_SCHEMA_UNSUPPORTED, None

    capsule_hash = capsule.get("capsule_hash")
    if not capsule_hash:
        return E010_CAPSULE_HASH_MISSING, None
    computed_hash = _canonical_capsule_hash(capsule)
    if capsule_hash.lower() != computed_hash.lower():
        return E011_CAPSULE_HASH_MISMATCH, None

    artifacts = capsule.get("artifacts", {})
    geom_entry = capsule.get("proofs", {}).get("geom", {})
    chunk_meta = capsule.get("chunk_meta") or {}
    row_index_ref = capsule.get("row_index_ref") or {}
    trace_spec_obj = capsule.get("trace_spec")
    trace_spec_hash = capsule.get("trace_spec_hash")
    if not trace_spec_obj or not trace_spec_hash:
        return E053_PROOF_STATEMENT_MISMATCH, None
    trace_spec = TraceSpecV1.from_obj(trace_spec_obj)
    computed_trace_spec_hash = compute_trace_spec_hash(trace_spec)
    if computed_trace_spec_hash.lower() != str(trace_spec_hash).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None

    policy_info = capsule.get("policy") or {}
    # Optional policy pinning
    policy_verified = False
    if policy_registry_root:
        expected_policy_hash = policy_info.get("policy_hash")
        if not expected_policy_hash:
            return E033_POLICY_HASH_MISMATCH, None
        try:
            actual_policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            if actual_policy_hash.lower() != expected_policy_hash.lower():
                return E033_POLICY_HASH_MISMATCH, None
            proof_data = json.loads(policy_proof_path.read_text())
            leaf_hash_bytes = bytes.fromhex(proof_data["leaf_hash"])
            if leaf_hash_bytes.hex() != actual_policy_hash:
                return E064_MERKLE_PROOF_INVALID, None
            siblings = [
                [bytes.fromhex(h) for h in level]
                for level in proof_data["siblings_by_level"]
            ]
            if not verify_kary(
                root=bytes.fromhex(policy_registry_root),
                leaf=leaf_hash_bytes,
                index=proof_data["leaf_index"],
                proof=siblings,
                arity=proof_data["arity"],
                total_leaves=proof_data["total_leaves"],
            ):
                return E034_POLICY_NOT_IN_REGISTRY, None
            policy_verified = True
        except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError, KeyError, ValueError):
            return E002_PARSE_FAILED, None

    # Proof artifact handling
    proof_path_entry = (
        geom_entry.get("path")
        or geom_entry.get("json_path")
        or geom_entry.get("bin_path")
        or artifacts.get("geom_proof")
    )
    if not proof_path_entry:
        return E050_PROOF_MISSING, None
    proof_path = _resolve(base, proof_path_entry)
    expected_hash = None
    for _fmt, meta in (geom_entry.get("formats") or {}).items():
        if meta.get("path") == str(proof_path_entry):
            expected_hash = meta.get("sha256_payload_hash")
            break
    if expected_hash:
        actual_hash = _compute_payload_hash(proof_path)
        if actual_hash.lower() != expected_hash.lower():
            return E052_PROOF_HASH_MISMATCH, None
    try:
        proof_encoding = encoding_for_path(manifest, base, proof_path) if manifest else None
        if proof_encoding == "dag_cbor_canonical_v1" or proof_path.suffix == ".bin":
            proof = proof_from_bytes(proof_path.read_bytes())
        else:
            proof = proof_from_json(proof_path.read_text())
    except Exception:
        return E002_PARSE_FAILED, None

    binding_status = _verify_row_commitment_binding(proof.row_commitment, chunk_meta, row_index_ref)
    if binding_status != OK:
        return binding_status, None

    statement_obj = capsule.get("statement")
    statement_hash_hex = capsule.get("statement_hash")
    if not statement_obj or not statement_hash_hex:
        return E053_PROOF_STATEMENT_MISMATCH, None
    statement = StatementV1.from_obj(statement_obj)
    computed_statement_hash = compute_statement_hash(statement)
    if computed_statement_hash.lower() != str(statement_hash_hex).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    if statement.trace_spec_hash and statement.trace_spec_hash.lower() != str(trace_spec_hash).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    policy_hash = policy_info.get("policy_hash")
    if policy_hash and statement.policy_hash.lower() != str(policy_hash).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    trace_root_expected = row_index_ref.get("commitment")
    if trace_root_expected and statement.trace_root.lower() != str(trace_root_expected).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    anchors_expected = []
    anchor_meta = capsule.get("anchor")
    if anchor_meta and anchor_meta.get("anchor_ref"):
        anchors_expected.append(anchor_meta)
    if anchors_expected and statement.anchors != anchors_expected:
        return E053_PROOF_STATEMENT_MISMATCH, None

    params = capsule["params"]
    geom_params = GeomAIRParams(
        steps=params["steps"],
        num_challenges=params["num_challenges"],
        r_challenges=params["r_challenges"],
        matrix=[[2, 1], [1, 1]],
    )
    init_state = GeomInitialState()
    vc = STCVectorCommitment(chunk_len=params["row_width"])
    statement_hash_bytes = bytes.fromhex(statement_hash_hex)
    geom_ok, verify_stats = zk_verify_geom(
        GEOM_PROGRAM,
        geom_params,
        init_state,
        vc,
        proof,
        statement_hash=statement_hash_bytes,
    )
    if not geom_ok:
        return E054_PROOF_VERIFICATION_FAILED, None

    # Optional Nova state check
    nova_stats_entry = artifacts.get("nova_stats")
    if nova_stats_entry:
        nova_stats_path = _resolve(base, str(nova_stats_entry))
        try:
            nova_stats = json.loads(nova_stats_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return E002_PARSE_FAILED, None
        expected_commitment = capsule.get("trace_commitment")
        if expected_commitment is not None and nova_stats.get("nova_state") != expected_commitment:
            return E055_NOVA_STATE_MISMATCH, None
    else:
        nova_stats_path = None

    # Row archive bookkeeping
    row_archive_path: Path | None = None
    row_archive_entry = artifacts.get("row_archive")
    if row_archive_entry:
        candidate: str | None = None
        if isinstance(row_archive_entry, dict):
            candidate = row_archive_entry.get("path") or row_archive_entry.get("abs_path")
            mode = row_archive_entry.get("mode", "")
            if not candidate and mode:
                return E062_ROW_ARCHIVE_MISSING, None
        else:
            candidate = str(row_archive_entry)
        if candidate:
            row_archive_path = _resolve(base, candidate)
            if not row_archive_path.exists():
                return E062_ROW_ARCHIVE_MISSING, None
    row_archive_info = (
        capsule.get("row_archive")
        or geom_entry.get("row_archive")
        or row_archive_entry
    )
    chunk_handles: list[str] = []
    chunk_tree_arity = 2
    chunk_roots: list[bytes] | None = None
    row_root_bytes: bytes | None = None
    if row_archive_info:
        chunk_handles = [str(h) for h in row_archive_info.get("chunk_handles", [])]
        chunk_tree_arity = int(row_archive_info.get("chunk_tree_arity") or 2)
        digest_hex = row_archive_info.get("chunk_roots_digest")
        if digest_hex:
            roots_path = row_archive_info.get("chunk_roots_bin_abs") or row_archive_info.get("chunk_roots_bin_path")
            if not roots_path:
                return E063_CHUNK_ROOTS_DIGEST_MISMATCH, None
            actual_digest = hashlib.sha256(_resolve(base, roots_path).read_bytes()).hexdigest()
            if actual_digest.lower() != digest_hex.lower():
                return E063_CHUNK_ROOTS_DIGEST_MISMATCH, None
        row_archive_abs = row_archive_info.get("abs_path")
        if row_archive_abs and not row_archive_path:
            row_archive_path = Path(row_archive_abs)
        if row_index_ref and row_index_ref.get("commitment"):
            chunk_roots, err = _load_chunk_roots(row_archive_info, base)
            if err != OK:
                return err, None
            arity = int(row_index_ref.get("tree_arity") or row_archive_info.get("chunk_tree_arity") or 2)
            levels = build_kary_levels(chunk_roots, arity)
            derived_root = root_from_levels(levels)
            if derived_root.hex() != row_index_ref.get("commitment"):
                return E064_MERKLE_PROOF_INVALID, None
            row_root_bytes = derived_root
            row_index_ok = True
        else:
            row_index_ok = False
    else:
        row_index_ok = False

    # Authorship + ACL
    acl, err = _load_acl(acl_path)
    if err != OK:
        return err, None
    require_authorship = bool(acl)
    auth_status, signer_hex, authorship_verified = _verify_authorship(capsule, capsule_hash, require_authorship)
    if auth_status != OK:
        return auth_status, None
    acl_status, acl_authorized = _verify_acl(
        (capsule.get("policy") or {}).get("policy_id"),
        signer_hex,
        acl,
    )
    if acl_status != OK:
        return acl_status, None

    # Data availability audit
    da_audit_verified = False
    da_policy = capsule.get("da_policy")
    legacy_profile = capsule.get("da_profile") if not da_policy else None
    if da_policy or legacy_profile:
        if not row_root_bytes or not chunk_roots:
            return E060_ROW_INDEX_COMMITMENT_MISSING, None
        chunk_meta = capsule.get("chunk_meta") or {}
        chunk_len = int(chunk_meta.get("chunk_len") or 0)
        provider_config: dict[str, Any] = {}
        timeout_ms = 0
        retry_count = 0
        if da_policy:
            sample_count = int(da_policy.get("k_samples") or 0)
            provider_config = da_policy.get("provider") or {}
            timeout_ms = int(da_policy.get("provider_timeout_ms") or 0)
            retry_count = int(da_policy.get("provider_retry_count") or 0)
            mode = (provider_config.get("mode") or "LOCAL_FILE").upper()
        else:
            sample_cfg = legacy_profile.get("sampling") or {}
            sample_count = int(sample_cfg.get("k_min") or 0)
            mode = (legacy_profile.get("mode") or "LIGHT_SAMPLING").upper()
        if mode not in {"LIGHT_SAMPLING", "LOCAL_FILE"}:
            return E070_DA_MODE_UNSUPPORTED, None
        archive_root = row_archive_path
        provider_root = provider_config.get("archive_root") if provider_config else None
        if provider_root:
            archive_root = Path(provider_root)
        if archive_root is None:
            return E062_ROW_ARCHIVE_MISSING, None
        provider = LocalFileSystemProvider(
            archive_root=archive_root,
            chunk_handles=chunk_handles,
            chunk_roots=chunk_roots,
            tree_arity=chunk_tree_arity,
        )
        wrapped_provider = (
            PolicyAwareDAClient(provider, retries=retry_count, timeout_ms=timeout_ms)
            if da_policy
            else provider
        )
        audit_status, da_audit_verified = _run_da_audit(
            capsule,
            chunk_roots,
            chunk_len,
            row_root_bytes,
            sample_count,
            wrapped_provider,
        )
        if audit_status != OK:
            return audit_status, None

    result = {
        "steps": params["steps"],
        "num_challenges": params["num_challenges"],
        "geom_verify_stats": verify_stats,
        "trace_commitment": capsule.get("trace_commitment"),
        "geom_proof_path": str(proof_path),
        "nova_stats_path": str(nova_stats_path) if nova_stats_path else None,
        "row_archive": str(row_archive_path) if row_archive_path else None,
        "capsule_hash_ok": True,
        "row_index_commitment_ok": row_index_ok,
        "policy_verified": policy_verified,
        "authorship_verified": authorship_verified,
        "acl_authorized": acl_authorized,
        "da_audit_verified": da_audit_verified,
    }
    return OK, result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_capsule(
    capsule_path: Path,
    *,
    policy_path: Path | None = None,
    policy_proof_path: Path | None = None,
    policy_registry_root: str | None = None,
    acl_path: Path | None = None,
) -> dict:
    status, result = _verify_capsule_core(
        capsule_path,
        policy_path=policy_path,
        policy_proof_path=policy_proof_path,
        policy_registry_root=policy_registry_root,
        acl_path=acl_path,
    )
    if status == OK:
        return result or {}
    if status == E062_ROW_ARCHIVE_MISSING:
        raise FileNotFoundError("row archive missing or unreadable")
    if status == E055_NOVA_STATE_MISMATCH:
        raise RuntimeError("Nova STC state mismatch between stats and capsule")
    raise ValueError(status)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a BEF strategy capsule")
    parser.add_argument("capsule", type=Path, help="path to strategy_capsule.(json|bin)")
    parser.add_argument("--policy", type=Path, help="path to policy file to verify")
    parser.add_argument("--policy-inclusion-proof", type=Path, help="path to JSON Merkle proof for policy")
    parser.add_argument("--policy-registry-root", type=str, help="hex Merkle root of trusted policy registry")
    parser.add_argument("--acl-path", type=Path, help="path to ACL JSON mapping policy IDs to authorized signer keys")
    args = parser.parse_args()

    policy_args = [args.policy, args.policy_registry_root, args.policy_inclusion_proof]
    if any(policy_args) and not all(policy_args):
        parser.error("--policy, --policy-inclusion-proof, and --policy-registry-root must be provided together")

    status, result = _verify_capsule_core(
        args.capsule,
        policy_path=args.policy,
        policy_proof_path=args.policy_inclusion_proof,
        policy_registry_root=args.policy_registry_root,
        acl_path=args.acl_path,
    )
    if status == OK:
        print(json.dumps(result, indent=2))
        sys.exit(0)
    error_payload = {"status": "REJECT", "error_code": status}
    print(json.dumps(error_payload, indent=2), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
