"""Simple chunk archive for streaming STC accumulators."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence


class ChunkArchive:
    """Disk-backed storage for chunk values (optional persistent root)."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="stc_chunks_")
            self.root = Path(self._tempdir.name)
            self._persistent = False
        else:
            self.root = Path(root_dir)
            self.root.mkdir(parents=True, exist_ok=True)
            self._tempdir = None
            self._persistent = True

    def store_chunk(self, chunk_index: int, values: Sequence[int]) -> str:
        path = self.root / f"chunk_{chunk_index}.json"
        path.write_text(json.dumps([int(v) for v in values]))
        rel = path.relative_to(self.root)
        return rel.as_posix()

    def load_chunk(self, handle: str) -> List[int]:
        path = Path(handle)
        if not path.is_absolute():
            path = self.root / path
        return [int(v) for v in json.loads(path.read_text())]

    def cleanup(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()


__all__ = ["ChunkArchive"]
