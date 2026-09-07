"""
Synaptic Memory (Hebbian / fast-weight prototype)
--------------------------------------------------
IMPORTANT SCIENTIFIC POSITION:
This is a simplified, conceptual/experimental Hebbian fast-weight memory
INSPIRED BY synaptic-plasticity ideas used in brain-inspired architectures
such as Dragon Hatchling (BDH). It is NOT a reimplementation of BDH and is
NOT claimed to be functionally equivalent to it. BDH's actual mechanism
(sparse, non-negative, neuron-synapse formulation with ReLU-low-rank
attention) is more elaborate; this module isolates one core idea from
that family — writing associations into a fixed-size synaptic matrix via
a Hebbian outer-product update — so the trade-off it produces (bounded
memory vs. interference) can be studied directly.

Update rule:
    W_t = lambda * W_{t-1} + eta * outer(v, k)

Where:
    W      = synaptic memory matrix, shape (dim, dim). FIXED SIZE.
    lambda = decay / retention factor  (0 < lambda <= 1)
    eta    = learning / plasticity rate
    k      = key vector
    v      = value vector

Retrieval:
    retrieved = W @ query
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SynapticMemory:
    dim: int
    decay: float = 0.99   # lambda
    lr: float = 1.0       # eta
    W: np.ndarray = None
    writes: int = 0

    def __post_init__(self):
        if self.W is None:
            self.W = np.zeros((self.dim, self.dim), dtype=np.float64)

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """Hebbian update. Matrix size never changes — this is the O(1)-space claim."""
        self.W = self.decay * self.W + self.lr * np.outer(value, key)
        self.writes += 1

    def write_batch(self, keys: np.ndarray, values: np.ndarray) -> None:
        for k, v in zip(keys, values):
            self.write(k, v)

    def raw_retrieve(self, query: np.ndarray) -> np.ndarray:
        """retrieved = W @ query (raw vector in value-space, not yet decoded)."""
        return self.W @ query

    def retrieve_from_candidates(
        self, query: np.ndarray, candidate_values: np.ndarray, candidate_labels: list[str]
    ) -> tuple[str, float, np.ndarray]:
        """
        Decode the raw retrieved vector against a known candidate set of
        value vectors (the controlled benchmark's vocabulary) via cosine
        similarity. Returns (best_label, similarity, raw_retrieved_vector).
        """
        raw = self.raw_retrieve(query)
        norm_raw = raw / (np.linalg.norm(raw) + 1e-9)
        Cn = candidate_values / (np.linalg.norm(candidate_values, axis=1, keepdims=True) + 1e-9)
        sims = Cn @ norm_raw
        best_idx = int(np.argmax(sims))
        return candidate_labels[best_idx], float(sims[best_idx]), raw

    def memory_bytes(self) -> int:
        """Measured memory footprint: fixed dim x dim float64 matrix. Does NOT grow with writes."""
        return self.dim * self.dim * 8

    def reset(self) -> None:
        self.W = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.writes = 0
