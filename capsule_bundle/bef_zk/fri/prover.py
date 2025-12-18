"""FRI prover using STC-backed vector commitments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import hashlib

from .config import FRIConfig
from .domain import fold_codeword
from .types import FRILayerInfo, FRIProof, FRILayerBatch
from bef_zk.stc.vc import VectorCommitment, VCCommitment

MODULUS = (1 << 61) - 1


def _mod(x: int) -> int:
    return x % MODULUS


def _derive_beta(root: bytes, round_idx: int) -> int:
    h = hashlib.sha256()
    h.update(root)
    h.update(round_idx.to_bytes(4, "big"))
    candidate = int.from_bytes(h.digest(), "big") % MODULUS
    if candidate == 0:
        candidate = 1
    return candidate


def _build_layers(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_evals: Sequence[int],
    base_commitment: VCCommitment,
) -> List[FRILayerInfo]:
    if len(base_evals) != fri_cfg.domain_size:
        raise ValueError("base evaluations length must match domain size")
    if fri_cfg.domain_size & (fri_cfg.domain_size - 1):
        raise ValueError("domain size must be a power of two")

    layers: List[FRILayerInfo] = []
    current = [int(v) % MODULUS for v in base_evals]
    commit = base_commitment

    for round_idx in range(fri_cfg.num_rounds):
        beta = _derive_beta(commit.root, round_idx)
        layers.append(FRILayerInfo(commitment=commit, beta=beta, length=len(current)))
        next_codeword = fold_codeword(current, beta, fri_cfg.field_modulus)
        commit = vc.commit(next_codeword)
        current = next_codeword

    # last layer commitment
    layers.append(FRILayerInfo(commitment=commit, beta=0, length=len(current)))
    return layers


def fri_prove(
    fri_cfg: FRIConfig,
    vc: VectorCommitment,
    base_evals: Sequence[int],
    base_commitment: VCCommitment,
    query_indices: Sequence[int],
) -> FRIProof:
    layers = _build_layers(fri_cfg, vc, base_evals, base_commitment)
    num_layers = len(layers)
    needed: List[set[int]] = [set() for _ in range(num_layers)]

    for idx in query_indices:
        cur_idx = idx
        for round_idx in range(fri_cfg.num_rounds):
            layer_info = layers[round_idx]
            if layer_info.length <= 1:
                raise ValueError("FRI layer length too small for folding")
            parent_idx = cur_idx // 2
            even_index = parent_idx * 2
            odd_index = even_index + 1
            needed[round_idx].add(even_index)
            needed[round_idx].add(odd_index)
            cur_idx = parent_idx
        needed[-1].add(cur_idx)

    batches: List[FRILayerBatch] = []
    for layer_idx, layer_info in enumerate(layers):
        idxs = sorted(needed[layer_idx])
        batch_proof = vc.open_batch(layer_info.commitment, idxs)
        batches.append(
            FRILayerBatch(
                layer_index=layer_idx,
                proof=batch_proof,
            )
        )

    proof = FRIProof(layers=layers, batches=batches)
    return proof
