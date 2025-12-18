"""Domain helpers for FRI (power-of-two)."""
from __future__ import annotations

from typing import List

FieldElem = int


def fold_codeword(values: List[FieldElem], alpha: FieldElem, modulus: int) -> List[FieldElem]:
    if len(values) % 2 != 0:
        raise ValueError("folding expects even-length codeword")
    out: List[FieldElem] = []
    for i in range(0, len(values), 2):
        a = values[i]
        b = values[i + 1]
        out.append((a + alpha * b) % modulus)
    return out


def next_index(idx: int) -> int:
    return idx // 2

