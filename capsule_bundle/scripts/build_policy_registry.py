#!/usr/bin/env python3
"""Build a Merkle tree over a directory of policy files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json

from bef_zk.stc.merkle import build_kary_levels, root_from_levels, prove_kary

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify a policy registry")
    parser.add_argument("policy_dir", type=Path, help="Directory containing policy files")
    parser.add_argument("--arity", type=int, default=16, help="Arity of the Merkle tree")
    args = parser.parse_args()

    if not args.policy_dir.is_dir():
        raise NotADirectoryError(f"Policy directory not found: {args.policy_dir}")

    policy_files = sorted([p for p in args.policy_dir.iterdir() if p.is_file() and not p.name.endswith(".proof")])
    if not policy_files:
        raise FileNotFoundError("No policy files found in directory")

    print(f"Found {len(policy_files)} policy files...")
    leaf_hashes = [hashlib.sha256(p.read_bytes()).digest() for p in policy_files]

    levels = build_kary_levels(leaf_hashes, args.arity)
    root = root_from_levels(levels)

    print(f"\nPolicy Registry Merkle Root (Arity={args.arity}):")
    print(root.hex())

    for i, policy_path in enumerate(policy_files):
        proof_siblings = prove_kary(levels, i, args.arity)
        proof_path = policy_path.with_suffix(policy_path.suffix + ".proof")
        proof_data = {
            "leaf_hash": leaf_hashes[i].hex(),
            "leaf_index": i,
            "total_leaves": len(leaf_hashes),
            "arity": args.arity,
            "siblings_by_level": [
                [p.hex() for p in level] for level in proof_siblings
            ]
        }
        proof_path.write_text(json.dumps(proof_data, indent=2))
        print(f"  - Wrote inclusion proof for {policy_path.name} to {proof_path.name}")

if __name__ == "__main__":
    main()