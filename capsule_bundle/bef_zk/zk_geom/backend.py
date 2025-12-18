"""Backends for committing / opening trace rows."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from bef_zk.stc.aok_cpu import MODULUS, ROOT_SEED, StreamingAccumulatorCPU
from bef_zk.stc.pc_backend import CHUNK_TREE_ARITY
from ..stc.merkle import build_levels, root_from_levels, prove as merkle_prove, verify as merkle_verify
from ..stc.pc_backend import STCChunkProof, stc_verify_chunk
from ..stc.vc import STCVectorCommitment, VCCommitment


FieldRow = List[int]


@dataclass
class RowCommitment:
    backend: str
    row_width: int
    params: Dict[str, Any]
    prover_state: Any | None = None  # cleared before serialization


@dataclass
class RowOpening:
    backend: str
    row_index: int
    row_values: FieldRow
    proof: Dict[str, Any]
    next_index: Optional[int] = None
    next_row_values: Optional[FieldRow] = None
    next_proof: Optional[Dict[str, Any]] = None


class RowBackend(Protocol):
    name: str

    def __init__(self, row_width: int, **kwargs: Any): ...

    def commit_rows(self, rows: List[FieldRow]) -> RowCommitment:
        """Commit to the full row matrix."""

    def open_row(self, commitment: RowCommitment, idx: int) -> tuple[FieldRow, Dict[str, Any]]:
        """Return (row_values, proof) for the specified row index."""

    def verify_leaf(
        self,
        commitment: RowCommitment,
        idx: int,
        row_values: FieldRow,
        proof: Dict[str, Any],
    ) -> bool:
        """Check that row_values belong to the commitment at index idx."""

    def streaming_init(self) -> Dict[str, Any]:  # pragma: no cover - interface default
        return {"rows": []}

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:  # pragma: no cover - interface default
        state.setdefault("rows", []).append(list(row))

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:  # pragma: no cover - interface default
        rows = state.get("rows", [])
        return self.commit_rows(rows)

    def streaming_init(self) -> Dict[str, Any]:
        """Return a mutable state used for streaming commits."""

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:
        """Append a row to the streaming state."""

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:
        """Finalize the streaming state into a RowCommitment."""


###############################################################################
# Backend registry helpers


_BACKENDS: Dict[str, type[RowBackend]] = {}


def register_backend(cls: type[RowBackend]) -> type[RowBackend]:
    _BACKENDS[cls.name] = cls
    return cls


def get_row_backend(name: str, row_width: int, **kwargs: Any) -> RowBackend:
    try:
        backend_cls = _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown row backend {name}") from exc
    return backend_cls(row_width=row_width, **kwargs)


def available_row_backends() -> List[str]:
    return sorted(_BACKENDS.keys())


###############################################################################
# STC streaming backend


@register_backend
class STCRowBackend(RowBackend):
    name = "geom_stc_fri"

    def __init__(self, row_width: int, archive_dir: str | Path | None = None, chunk_tree_arity: int | None = None):
        self.row_width = row_width
        self.archive_dir = Path(archive_dir).resolve() if archive_dir is not None else None
        self.chunk_tree_arity = chunk_tree_arity

    def commit_rows(self, rows: List[FieldRow]) -> RowCommitment:
        flat: List[int] = []
        for row in rows:
            if len(row) != self.row_width:
                raise ValueError("row width mismatch")
            flat.extend(int(v) for v in row)
        vc = STCVectorCommitment(
            chunk_len=self.row_width,
            archive_dir=self.archive_dir,
        )
        if self.chunk_tree_arity is not None:
            vc.chunk_tree_arity = self.chunk_tree_arity
        vc_commit = vc.commit(flat)
        store = vc._store.get(vc_commit.root, {})
        chunk_records = store.get("chunks", [])
        params = {
            "root": vc_commit.root.hex(),
            "length": vc_commit.length,
            "chunk_len": vc_commit.chunk_len,
            "num_chunks": vc_commit.num_chunks,
            "chain_root": vc_commit.chain_root.hex(),
            "challenges": list(vc_commit.challenges),
            "sketches": [int(x) % MODULUS for x in vc_commit.sketches],
            "powers": [int(x) % MODULUS for x in vc_commit.powers],
            "archive_root": vc.get_archive_root(vc_commit),
            "chunk_handles": vc.get_chunk_handles(vc_commit),
            "chunk_tree_arity": getattr(vc_commit, "chunk_tree_arity", CHUNK_TREE_ARITY),
            "chunk_roots_hex": [getattr(chunk, "root").hex() for chunk in chunk_records],
        }
        return RowCommitment(
            backend=self.name,
            row_width=self.row_width,
            params=params,
            prover_state={"vc": vc, "commitment": vc_commit},
        )

    def open_row(self, commitment: RowCommitment, idx: int) -> tuple[FieldRow, Dict[str, Any]]:
        state = commitment.prover_state
        if state is None:
            raise ValueError("prover state missing for row openings")
        vc: STCVectorCommitment = state["vc"]
        vc_commit: VCCommitment = state["commitment"]
        chunk_proof = vc.open_chunk(vc_commit, idx)
        return list(chunk_proof.values), chunk_proof.to_json()

    def verify_leaf(
        self,
        commitment: RowCommitment,
        idx: int,
        row_values: FieldRow,
        proof: Dict[str, Any],
    ) -> bool:
        params = commitment.params
        vc_commit = VCCommitment(
            root=bytes.fromhex(params["root"]),
            length=int(params["length"]),
            chunk_len=int(params["chunk_len"]),
            num_chunks=int(params["num_chunks"]),
            chain_root=bytes.fromhex(params.get("chain_root", ROOT_SEED.hex())),
            challenges=list(params.get("challenges", [])),
            sketches=[int(x) % MODULUS for x in params.get("sketches", [])],
            powers=[int(x) % MODULUS for x in params.get("powers", [])],
            chunk_tree_arity=int(params.get("chunk_tree_arity", CHUNK_TREE_ARITY)),
        )
        chunk_proof = STCChunkProof.from_json(proof)
        if chunk_proof.chunk_index != idx:
            return False
        if not stc_verify_chunk(vc_commit, chunk_proof):
            return False
        expected = [int(v) for v in row_values]
        return all((chunk_proof.values[i] % MODULUS) == (expected[i] % MODULUS) for i in range(len(expected)))

    def streaming_init(self) -> Dict[str, Any]:
        return {
            "rows": [],
            "acc": StreamingAccumulatorCPU(num_challenges=0, chunk_len=self.row_width),
        }

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:
        if len(row) != self.row_width:
            raise ValueError("row width mismatch")
        state["rows"].append(list(row))
        state["acc"].add_chunk(row)

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:
        rows = state.get("rows", [])
        return self.commit_rows(rows)


###############################################################################
# Plain Merkle backend


def _hash_row(row_index: int, row: FieldRow) -> bytes:
    h = hashlib.sha256()
    h.update(row_index.to_bytes(8, "big"))
    for val in row:
        h.update(int(val).to_bytes(16, "big", signed=False))
    return h.digest()


@register_backend
class MerkleRowBackend(RowBackend):
    name = "geom_plain_fri"

    def __init__(self, row_width: int, **_: Any):
        self.row_width = row_width

    def commit_rows(self, rows: List[FieldRow]) -> RowCommitment:
        leaves = [_hash_row(idx, row) for idx, row in enumerate(rows)]
        levels = build_levels(leaves)
        root = root_from_levels(levels)
        params = {
            "root": root.hex(),
            "num_rows": len(rows),
        }
        return RowCommitment(
            backend=self.name,
            row_width=self.row_width,
            params=params,
            prover_state={"rows": rows, "levels": levels},
        )

    def open_row(self, commitment: RowCommitment, idx: int) -> tuple[FieldRow, Dict[str, Any]]:
        state = commitment.prover_state
        if state is None:
            raise ValueError("prover state missing for row openings")
        rows = state["rows"]
        levels = state["levels"]
        if idx < 0 or idx >= len(rows):
            raise IndexError("row index out of range")
        row = rows[idx]
        path = [hx.hex() for hx in merkle_prove(levels, idx)]
        return list(row), {"path": path}

    def verify_leaf(
        self,
        commitment: RowCommitment,
        idx: int,
        row_values: FieldRow,
        proof: Dict[str, Any],
    ) -> bool:
        if idx < 0 or idx >= int(commitment.params.get("num_rows", 0)):
            return False
        if len(row_values) != commitment.row_width:
            return False
        root = bytes.fromhex(commitment.params["root"])
        path = [bytes.fromhex(x) for x in proof.get("path", [])]
        leaf = _hash_row(idx, row_values)
        return merkle_verify(root, leaf, idx, path)

    def streaming_init(self) -> Dict[str, Any]:
        return {
            "rows": [],
            "leaves": [],
        }

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:
        if len(row) != self.row_width:
            raise ValueError("row width mismatch")
        idx = len(state["rows"])
        state["rows"].append(list(row))
        state["leaves"].append(_hash_row(idx, row))

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:
        rows: List[FieldRow] = state.get("rows", [])
        leaves: List[bytes] = state.get("leaves", [])
        if len(leaves) != len(rows):
            leaves = [_hash_row(idx, row) for idx, row in enumerate(rows)]
        levels = build_levels(leaves)
        root = root_from_levels(levels)
        params = {
            "root": root.hex(),
            "num_rows": len(rows),
        }
        # Store copies so future openings can be derived.
        stored_rows = [list(row) for row in rows]
        commitment = RowCommitment(
            backend=self.name,
            row_width=self.row_width,
            params=params,
            prover_state={"rows": stored_rows, "levels": levels},
        )
        return commitment
