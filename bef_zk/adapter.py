"""Abstractions for plugging alternative trace/proof backends into the pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .spec import TraceSpecV1


@dataclass
class TraceArtifacts:
    """Normalized outputs from TraceAdapter.simulate_trace."""

    trace_id: str
    trace_spec: TraceSpecV1
    trace_spec_hash: str
    bef_trace: dict[str, Any]
    row_width: int
    context: Any
    trace_time_sec: Optional[float] = None


@dataclass
class TraceCommitment:
    """Outputs from the TraceAdapter.commit_to_trace phase."""

    row_commitment: Any
    row_archive_artifact: dict[str, Any]
    chunk_handles: list[Any]
    chunk_roots_hex: list[str]
    chunk_roots_digest: str
    chunk_roots_paths: dict[str, Path]
    profile_data: dict[str, float]
    extra: dict[str, Any] | None = None
    backend_state: Any | None = None


@dataclass
class ProofArtifacts:
    """Outputs from TraceAdapter.generate_proof used by the pipeline."""

    proof_obj: Any
    proof_json: str
    proof_bytes: bytes
    profile_data: dict[str, float]
    chunk_leaf_stats: dict[str, int]
    extra: dict[str, Any] | None = None


class TraceAdapter(ABC):
    """Interface for pluggable trace/proof backends."""

    name: str = "unknown"

    def __init__(self, args: Any) -> None:
        self.args = args
        self._progress_callback = None

    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        """Hook for adapter-specific CLI arguments."""

    @abstractmethod
    def simulate_trace(self, args: Any) -> TraceArtifacts:
        """Produce a trace along with metadata required by the pipeline."""

    @abstractmethod
    def extract_public_inputs(self, artifacts: TraceArtifacts) -> list[dict[str, Any]]:
        """Return the public inputs derived from the prepared trace."""

    @abstractmethod
    def commit_to_trace(
        self,
        artifacts: TraceArtifacts,
        *,
        row_archive_dir: Path,
    ) -> TraceCommitment:
        """Commit to the execution trace and export STC artifacts."""

    @abstractmethod
    def generate_proof(
        self,
        artifacts: TraceArtifacts,
        commitment: TraceCommitment,
        *,
        statement_hash: bytes,
        encoding_id: str,
        trace_path: Path,
    ) -> ProofArtifacts:
        """Generate the backend proof using a finalized statement hash."""

    @abstractmethod
    def verify(
        self,
        proof_json: str,
        statement_hash: bytes,
        artifacts: TraceArtifacts,
    ) -> tuple[bool, dict[str, Any], float]:
        """Run the backend verifier for sanity checks and benchmarking."""

    def set_progress_callback(self, callback: Optional[Callable[[dict[str, Any]], None]]) -> None:
        """Optional hook used by CapsuleBench to observe proving progress."""

        self._progress_callback = callback

    def _emit_progress(self, event_type: str, data: dict[str, Any]) -> None:
        if self._progress_callback is not None:
            self._progress_callback({"type": event_type, "data": data})
