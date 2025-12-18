"""Collect host environment manifests for capsule-bench runs."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _hash_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _maybe_run_command(cmd: list[str]) -> Dict[str, Any]:
    if not cmd:
        return {"command": [], "error": "empty command"}
    if shutil.which(cmd[0]) is None:
        return {
            "command": cmd,
            "error": "command unavailable",
        }
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - best effort
        return {"command": cmd, "error": str(exc)}
    return {
        "command": cmd,
        "stdout": out.decode("utf-8", errors="replace").strip(),
    }


def _detect_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages) * int(page_size)
        except (OSError, ValueError):  # pragma: no cover - platform dependent
            return None
    return None


def _detect_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"schema": "bef_gpu_manifest_v1", "detected": False}
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    result = _maybe_run_command(cmd)
    if "stdout" in result and result["stdout"]:
        devices = []
        for line in result["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts:
                continue
            entry = {"model": parts[0]}
            if len(parts) > 1:
                entry["driver_version"] = parts[1]
            devices.append(entry)
        info["detected"] = bool(devices)
        info["devices"] = devices
    else:
        info["error"] = result.get("error", "gpu info unavailable")
    return info


def _hardware_manifest() -> Dict[str, Any]:
    return {
        "schema": "bef_hardware_manifest_v1",
        "generated_at": _now_iso(),
        "cpu": {
            "model": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(),
        },
        "memory_bytes": _detect_memory_bytes(),
        "gpu": _detect_gpu_info(),
    }


def _os_manifest() -> Dict[str, Any]:
    return {
        "schema": "bef_os_fingerprint_v1",
        "generated_at": _now_iso(),
        "platform": platform.platform(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": sys.version,
    }


def _toolchain_manifest() -> Dict[str, Any]:
    rustc = _maybe_run_command(["rustc", "--version"])
    cargo = _maybe_run_command(["cargo", "--version"])
    git = _maybe_run_command(["git", "rev-parse", "HEAD"])
    return {
        "schema": "bef_toolchain_manifest_v1",
        "generated_at": _now_iso(),
        "python": sys.version,
        "rustc_version": rustc.get("stdout"),
        "cargo_version": cargo.get("stdout"),
        "git_commit": git.get("stdout"),
        "commands": {
            "rustc": rustc,
            "cargo": cargo,
            "git": git,
        },
    }


@dataclass
class ManifestBundle:
    base_dir: Path
    files: dict[str, Path]
    hashes: dict[str, str]
    anchor_ref: str


def collect_manifests(output_dir: Path) -> ManifestBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "hardware_manifest": _hardware_manifest(),
        "os_fingerprint": _os_manifest(),
        "toolchain_manifest": _toolchain_manifest(),
    }
    files: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, payload in manifests.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        files[name] = path
        hashes[name] = f"sha256:{_hash_file(path)}"
    entries = []
    for name, path in files.items():
        entries.append({"name": name, "path": str(path), "sha256": hashes.get(name)})
    index = {
        "schema": "bef_manifest_index_v1",
        "generated_at": _now_iso(),
        "entries": entries,
    }
    index_path = output_dir / "manifest_index.json"
    index_path.write_text(json.dumps(index, indent=2))
    files["manifest_index"] = index_path
    hashes["manifest_index"] = f"sha256:{_hash_file(index_path)}"
    anchor_payload = json.dumps({"schema": "capsule_bench_manifest_anchor_v1", "hashes": hashes}, sort_keys=True).encode()
    anchor_ref = f"capsulebench_manifest_v1:{_hash_bytes(anchor_payload)}"
    return ManifestBundle(base_dir=output_dir, files=files, hashes=hashes, anchor_ref=anchor_ref)
