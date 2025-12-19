"""capsule-bench CLI."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import click

from .manifests import ManifestBundle, collect_manifests
from .packing import create_capsulepack

ROOT = Path(__file__).resolve().parents[1]
RUN_META_SCHEMA = "capsule_bench_run_meta_v1"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")



def _compute_anchor(bundle: ManifestBundle) -> str:
    return bundle.anchor_ref


def _hash_policy(policy_path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    hasher.update(policy_path.read_bytes())
    return hasher.hexdigest()


def _run_pipeline(
    pipeline_args: Tuple[str, ...],
    *,
    backend: str,
    policy_path: Path,
    policy_id: str,
    policy_version: str,
    output_dir: Path,
    anchor_ref: str,
    trace_id: str,
    track_id: str,
    docker_image_digest: str | None,
    events_log: Path | None,
    private_key: Path | None,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--backend",
        backend,
        "--policy",
        str(policy_path),
        "--policy-id",
        policy_id,
        "--policy-version",
        policy_version,
        "--output-dir",
        str(output_dir),
        "--trace-id",
        trace_id,
        "--anchor-ref",
        anchor_ref,
        "--anchor-rule-id",
        "capsule_bench_manifest_v1",
        "--track-id",
        track_id,
    ]
    if docker_image_digest:
        cmd.extend(["--docker-image-digest", docker_image_digest])
    if events_log:
        cmd.extend(["--events-log", str(events_log)])
    if private_key:
        cmd.extend(["--private-key", str(private_key)])
    cmd.extend(pipeline_args)
    subprocess.run(cmd, check=True)


def _load_capsule_path(run_dir: Path) -> Path:
    pipeline_dir = run_dir / "pipeline"
    return pipeline_dir / "strategy_capsule.json"


@click.group()
def cli() -> None:
    """capsule-bench CLI entrypoint."""


@cli.command("run")
@click.option("--backend", default="geom", help="Trace adapter backend id.")
@click.option("--policy", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--policy-id", required=True, help="Policy identifier embedded in the capsule.")
@click.option("--policy-version", default="unspecified")
@click.option("--track-id", required=True, help="Benchmark track identifier (policy track).")
@click.option("--docker-image-digest", type=str, help="Docker image digest used for deterministic builds.")
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("out/capsule_runs"))
@click.option("--run-id", type=str, help="Custom run identifier (default: timestamp slug).")
@click.option("--trace-id", type=str, help="Optional trace id override.")
@click.option(
    "--private-key",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to secp256k1 private key used to sign the capsule.",
)
@click.argument("pipeline_args", nargs=-1, type=str)
def run_command(
    backend: str,
    policy: Path,
    policy_id: str,
    policy_version: str,
    track_id: str,
    docker_image_digest: str | None,
    output_root: Path,
    run_id: str | None,
    trace_id: str | None,
    private_key: Path | None,
    pipeline_args: Tuple[str, ...],
) -> None:
    """Execute the prover pipeline and capture manifests."""

    run_id = run_id or _timestamp_slug()
    trace_id = trace_id or run_id
    run_dir = (output_root / run_id).resolve()
    pipeline_dir = run_dir / "pipeline"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_root = run_dir / "manifests"
    manifest_bundle = collect_manifests(manifest_root)
    anchor_ref = _compute_anchor(manifest_bundle)
    policy_copy = run_dir / "policy.json"
    shutil.copy2(policy, policy_copy)
    events_path = run_dir / "events.jsonl"

    try:
        _run_pipeline(
            pipeline_args,
            backend=backend,
            policy_path=policy_copy,
            policy_id=policy_id,
            policy_version=policy_version,
            output_dir=pipeline_dir,
            anchor_ref=anchor_ref,
            trace_id=trace_id,
            track_id=track_id,
            docker_image_digest=docker_image_digest,
            events_log=events_path,
            private_key=private_key,
        )
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"pipeline execution failed: {exc}") from exc

    capsule_path = _load_capsule_path(run_dir)
    run_meta = {
        "schema": RUN_META_SCHEMA,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "backend": backend,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "policy_hash": _hash_policy(policy_copy),
        "policy_copy": str(policy_copy),
        "pipeline_output": str(pipeline_dir),
        "capsule_path": str(capsule_path),
        "manifests_root": str(manifest_bundle.base_dir),
        "manifest_hashes": manifest_bundle.hashes,
        "anchor_ref": anchor_ref,
        "events_path": str(events_path),
        "manifest_root": str(manifest_root),
        "track_id": track_id,
        "docker_image_digest": docker_image_digest,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    click.echo(f"capsule-bench run completed: {run_dir}")


@cli.command("pack")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--pack-name", type=str, help="Override output tgz name (without extension).")
def pack_command(run_dir: Path, pack_name: str | None) -> None:
    """Assemble capsulepack.tgz from a previous run."""

    run_meta_path = run_dir / "run_meta.json"
    if not run_meta_path.exists():
        raise click.ClickException(f"missing run_meta.json in {run_dir}")
    pack_dir, tar_path = create_capsulepack(run_meta_path, pack_name=pack_name)
    click.echo(f"capsulepack assembled at {pack_dir}")
    click.echo(f"archive written to {tar_path}")


def main() -> None:
    cli(prog_name="capsule-bench")


if __name__ == "__main__":
    main()
