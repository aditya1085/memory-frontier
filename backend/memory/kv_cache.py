"""
KV Cache Baseline
-----------------
Standard Transformer-style memory: every (key, value) pair ever written
is stored explicitly. Memory footprint grows linearly with the number
of facts written (O(N)).

Retrieval is done via cosine similarity between the query vector and
every stored key (brute-force, same as attention's QK^T comparison).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class KVCache:
    dim: int
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append a new (key, value) pair. Memory grows by one slot."""
        self.keys.append(key.astype(np.float64))
        self.values.append(value.astype(np.float64))

    def write_batch(self, keys: np.ndarray, values: np.ndarray) -> None:
        for k, v in zip(keys, values):
            self.write(k, v)

    def retrieve(self, query: np.ndarray) -> tuple[np.ndarray, float, int]:
        """
        Return (retrieved_value, similarity_score, num_comparisons).
        Brute-force cosine similarity search over all stored keys —
        this is what a growing KV cache / full attention has to do.
        """
        if not self.keys:
            return np.zeros(self.dim), 0.0, 0

        K = np.stack(self.keys)  # (N, dim)
        q = query / (np.linalg.norm(query) + 1e-9)
        Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
        sims = Kn @ q  # (N,)
        best_idx = int(np.argmax(sims))
        return self.values[best_idx], float(sims[best_idx]), len(self.keys)

    def num_entries(self) -> int:
        return len(self.keys)

    def memory_bytes(self) -> int:
        """Measured memory footprint: 2 float64 vectors per stored fact."""
        return self.num_entries() * self.dim * 8 * 2

    def reset(self) -> None:
        self.keys.clear()
        self.values.clear()
