from __future__ import annotations

import json
from pathlib import Path

from capsule_bench.manifests import collect_manifests
from capsule_bench.packing import write_pack_meta


def test_collect_manifests_creates_files(tmp_path: Path) -> None:
    bundle = collect_manifests(tmp_path)
    assert bundle.base_dir == tmp_path
    # hardware/os/toolchain manifests should exist
    for key in ["hardware_manifest", "os_fingerprint", "toolchain_manifest"]:
        assert key in bundle.files
        path = bundle.files[key]
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["schema"].startswith("bef_")
    assert bundle.anchor_ref.startswith("capsulebench_manifest_v1:")


def test_write_pack_meta(tmp_path: Path) -> None:
    pack_dir = tmp_path / "capsulepack"
    pack_dir.mkdir()
    (pack_dir / "capsule.json").write_text("{}")
    (pack_dir / "policy.json").write_text("policy")
    meta = write_pack_meta(pack_dir)
    pack_meta_path = pack_dir / "pack_meta.json"
    assert pack_meta_path.exists()
    recorded = json.loads(pack_meta_path.read_text())
    assert recorded["schema"] == "capsulepack_meta_v1"
    entries = {entry["path"] for entry in meta["entries"]}
    assert "capsule.json" in entries
    assert "policy.json" in entries
