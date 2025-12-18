"""RISC Zero TraceAdapter implementation leveraging journal binding."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bef_zk.adapter import (
    ProofArtifacts,
    TraceAdapter,
    TraceArtifacts,
    TraceCommitment,
)
from bef_zk.spec import TraceSpecV1, compute_trace_spec_hash

try:  # pragma: no cover - optional dependency
    from risc0_zkvm.executor import Executor
    from risc0_zkvm.session import Session
    from risc0_zkvm.prover import Prover
except ImportError:  # pragma: no cover - binder not available in CI
    Executor = None
    Session = None
    Prover = None
    _HAVE_RISC0 = False
else:  # pragma: no cover - exercised only when RISC0 runtime is installed
    _HAVE_RISC0 = True


def _ensure_risc0() -> None:
    if not _HAVE_RISC0:
        raise RuntimeError(
            "RISC Zero python bindings are not available. Install risc0-zkvm to use this adapter.")


@dataclass
class Risc0TraceCommitment(TraceCommitment):
    """Trace commitment wrapper that holds the RISC Zero Session."""

    session: Session | None = None


class Risc0TraceAdapter(TraceAdapter):
    name = "risc0"

    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        parser.add_argument(
            "--run-config",
            type=str,
            help="JSON blob describing ELF path and optional private inputs",
        )

    def simulate_trace(self, args: Any) -> TraceArtifacts:
        if not args.run_config:
            raise ValueError("--run-config is required for the Risc0TraceAdapter")
        try:
            if Path(args.run_config).exists():
                run_cfg = json.loads(Path(args.run_config).read_text())
            else:
                run_cfg = json.loads(args.run_config)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid config
            raise ValueError("invalid JSON provided to --run-config") from exc

        elf_path = run_cfg.get("elf_path")
        if not elf_path:
            raise ValueError("run-config must include an 'elf_path'")

        trace_spec = TraceSpecV1(
            spec_version="1.0",
            trace_format_id="RISC0_JOURNAL_V1",
            record_schema_ref="inline",
            encoding_id=args.encoding_id,
            field_modulus_id="risc0_babybear",
        )
        trace_spec_hash = compute_trace_spec_hash(trace_spec)
        bef_trace = {"backend": "risc0", "elf_path": elf_path}
        context = {
            "run_config": run_cfg,
        }
        return TraceArtifacts(
            trace_id=args.trace_id,
            trace_spec=trace_spec,
            trace_spec_hash=trace_spec_hash,
            bef_trace=bef_trace,
            row_width=0,
            context=context,
        )

    def extract_public_inputs(self, artifacts: TraceArtifacts) -> list[dict[str, Any]]:
        # RISC Zero proofs expose all public data via the journal; the CapsuleBench statement
        # simply declares the absence of structured public inputs for now.
        return []

    def commit_to_trace(
        self,
        artifacts: TraceArtifacts,
        *,
        row_archive_dir: Path,
    ) -> Risc0TraceCommitment:
        _ensure_risc0()
        run_cfg = artifacts.context.get("run_config", {})
        elf_path = run_cfg.get("elf_path")
        private_input = run_cfg.get("private_input")
        if not elf_path:
            raise ValueError("run-config missing 'elf_path'")

        start = time.perf_counter()
        executor = Executor(elf_path)
        if private_input:
            executor.add_input(private_input)
        session = executor.run()
        elapsed = time.perf_counter() - start

        journal_bytes = getattr(session, "journal", b"") or b""
        root_hex = hashlib.sha256(journal_bytes).hexdigest()

        row_archive_dir.mkdir(parents=True, exist_ok=True)
        chunk_roots_json = row_archive_dir / "chunk_roots.json"
        chunk_roots_json.write_text("[]")
        chunk_roots_bin = row_archive_dir / "chunk_roots.bin"
        chunk_roots_bin.write_bytes(b"")
        chunk_roots_digest = hashlib.sha256(chunk_roots_bin.read_bytes()).hexdigest()

        row_commitment = SimpleNamespace(
            backend="risc0_session_v1",
            row_width=0,
            params={
                "root": root_hex,
                "chunk_tree_arity": 1,
                "chunk_len": 0,
                "num_chunks": 0,
            },
        )
        chunk_handles: list[str] = []
        chunk_roots_paths = {"json": chunk_roots_json, "bin": chunk_roots_bin}
        row_archive_artifact = {
            "mode": "INLINE",
            "abs_path": str(row_archive_dir),
            "chunk_roots_path": str(chunk_roots_json),
            "chunk_roots_bin_path": str(chunk_roots_bin),
            "chunk_roots_format": "hex_json_v1",
            "chunk_roots_bin_format": "raw32_v1",
            "chunk_roots_digest": chunk_roots_digest,
        }

        return Risc0TraceCommitment(
            row_commitment=row_commitment,
            row_archive_artifact=row_archive_artifact,
            chunk_handles=chunk_handles,
            chunk_roots_hex=[],
            chunk_roots_digest=chunk_roots_digest,
            chunk_roots_paths=chunk_roots_paths,
            profile_data={"time_session_sec": elapsed},
            extra={"journal_len": len(journal_bytes)},
            session=session,
        )

    def generate_proof(
        self,
        artifacts: TraceArtifacts,
        commitment: Risc0TraceCommitment,
        *,
        statement_hash: bytes,
        encoding_id: str,
        trace_path: Path,
    ) -> ProofArtifacts:
        _ensure_risc0()
        if not isinstance(commitment, Risc0TraceCommitment):
            raise TypeError("generate_proof expected a Risc0TraceCommitment")
        if commitment.session is None:
            raise RuntimeError("Risc0TraceCommitment missing Session data")

        start = time.perf_counter()
        prover = Prover(commitment.session)
        receipt = prover.run()
        elapsed = time.perf_counter() - start

        journal_bytes = getattr(receipt, "journal", b"") or b""
        if journal_bytes != statement_hash:
            raise RuntimeError(
                "RISC Zero journal binding failed: guest did not commit the provided statement hash"
            )

        proof_json = receipt.to_json()
        proof_bytes = proof_json.encode("utf-8")
        profile_data = {
            **commitment.profile_data,
            "time_prove_sec": elapsed,
        }
        chunk_stats = {
            "leaf_auth_bytes_baseline": 0,
            "leaf_auth_bytes_actual": 0,
            "leaf_auth_bytes_saved": 0,
            "leaf_auth_savings_ratio": 0,
        }
        return ProofArtifacts(
            proof_obj=receipt,
            proof_json=proof_json,
            proof_bytes=proof_bytes,
            profile_data=profile_data,
            chunk_leaf_stats=chunk_stats,
            extra=None,
        )

    def verify(
        self,
        proof_json: str,
        statement_hash: bytes,
        artifacts: TraceArtifacts,
    ) -> tuple[bool, dict[str, Any], float]:
        _ensure_risc0()
        start = time.perf_counter()
        # Late import to avoid hitting optional dependency unless needed.
        from risc0_zkvm.receipt import Receipt  # pragma: no cover

        receipt = Receipt.from_json(proof_json)
        ok = receipt.verify()
        journal = getattr(receipt, "journal", b"") or b""
        ok = ok and journal == statement_hash
        elapsed = time.perf_counter() - start
        stats = {"journal_len": len(journal)}
        return ok, stats, elapsed
