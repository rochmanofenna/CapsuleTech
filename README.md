# STC / Geometry Backend

Hash-only STC + FRI pipeline for streaming traces, including TraceSpecV1/StatementV1 bindings, policy registry pinning, and a policy-aware DA audit.

## Key Features

- STC-backed vector + polynomial commitments
- TraceSpecV1: canonical, content-addressed trace spec stored in every capsule (`trace_spec`, `trace_spec_hash`)
- StatementV1: binds trace spec, policy hash, trace root, public inputs, anchors; hashed and fed to the STARK transcript (`statement`, `statement_hash`)
- Policy registry tooling (`scripts/build_policy_registry.py`) and policy-aware DA audit (retry/timeout + E074 availability codes)
- Authorship (secp256k1) and ACL enforcement

## Repo Layout

```
BEF/
├── bef_zk/          # AIR, STC backends, codec, TraceSpec/Statement helpers
├── scripts/         # prover, verifier, policy tooling, DA sampling
├── docs/            # backend architecture, DA protocol, TraceSpec docs
├── tests/           # regression/unit tests (verifier, DA audit)
└── bench/           # benchmarking harnesses
```

## TraceSpecV1 & StatementV1

See `docs/trace_statement_spec.md` for serialization + hashing details.

- `trace_spec`: describes trace format (schema ref, encoding, field modulus)
- `statement`: binds trace spec hash, policy hash, trace root, public inputs, anchors, optional external commitments
- `trace_spec_hash` / `statement_hash`: canonical CBOR hashes used both in the capsule and in the STARK transcript

## Running the Pipeline

```
PYTHONPATH=. .venv/bin/python scripts/run_pipeline.py --backend geom \
    --steps 4096 --num-challenges 4 --num-queries 32 \
    --output-dir out/geom_demo --trace-id geom_demo \
    --policy policies/policy1.txt --policy-id rollup_policy --policy-version v1
```

Outputs:

- `out/.../geom_proof.{json,bin}` – STARK proof (includes row commitment metadata)
- `out/.../strategy_capsule.{json,bin}` – capsule with trace spec, statement, policy hash, DA policy
- `out/.../artifact_manifest.json` – content-addressed artifact index
- `out/.../row_archive/` – STC chunk archive, Merkle roots/digest

## Verifying a Capsule

```
PYTHONPATH=. .venv/bin/python scripts/verify_capsule.py out/geom_demo/strategy_capsule.json \
    --policy policies/benchmark_policy_v1.json \
    --manifest-root out/capsule_runs/<run_id>/manifests
```

- Checks capsule hash, trace spec hash, statement hash, policy registry, proof payload hashes, row commitment, authorship/ACL, Nova state (if present), and DA audit (policy-aware retries/timeouts).
- Returns JSON:
  - Success: proof stats, DA audit flag, etc.
  - Failure: `{"status": "REJECT", "error_code": "E0xx"}`

## Tests

```
PYTHONPATH=. .venv/bin/python -m pytest tests/test_da_provider.py
PYTHONPATH=. .venv/bin/python -m pytest tests/test_capsule_verify.py -k da --maxfail=1
```

## Docs

- `docs/stc_backend_architecture.md` – STC as VC/PC backend
- `docs/trace_statement_spec.md` – TraceSpecV1/StatementV1 spec
- `docs/roadmap.md` – consolidated roadmap for TraceAdapters, DA providers, and policy tooling
- `docs/benchmark_policy_schema.json` – JSON schema for `bef_benchmark_policy_v1`
- `docs/hssa_da_protocol.md` – DA sampling protocol and guarantees
- `docs/stc_da_profiles.md` – DA policy profiles
- `server/README.md` – FastAPI relay for CapsuleBench live event streams

## CapsuleBench CLI

The `capsule-bench` CLI wraps the pipeline, captures hardware/toolchain manifests, and assembles
the canonical `capsulepack.tgz` artifact.

```
PYTHONPATH=. .venv/bin/python scripts/capsule_bench.py run \
    --backend geom \
    --policy policies/benchmark_policy_v1.json \
    --policy-id baseline_policy_v1 \
    --track-id baseline_no_accel \
    --docker-image-digest sha256:<digest>

PYTHONPATH=. .venv/bin/python scripts/capsule_bench.py pack --run-dir out/capsule_runs/run_YYYYMMDD_HHMMSS
```

`run` captures manifests and executes `scripts/run_pipeline.py`; `pack` enforces the canonical
capsulepack layout and writes `<run_id>.capsulepack.tgz`.

## Benchmarks

See `bench/` scripts (e.g., `bench/bench_geom_stc.py`, `bench/bench_streaming_backends.py`) for commitment throughput and STC vs KZG back-ends.
