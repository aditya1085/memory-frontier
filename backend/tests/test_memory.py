"""
Unit tests for the core memory models. Run with: pytest tests/
These tests check REAL properties of the implementation:
- KV cache memory grows linearly with writes
- Synaptic memory footprint stays constant regardless of writes
- KV cache achieves perfect recall on a small, non-colliding fact set
- Synaptic memory recall degrades as more facts are written (interference)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.kv_cache import KVCache
from memory.synaptic_memory import SynapticMemory
from memory.evaluator import run_associative_recall, retention_curve, generate_fact_set


def test_kv_cache_memory_grows_linearly():
    kv = KVCache(dim=32)
    sizes = []
    keys, values, _ = generate_fact_set(50, 32, seed=1)
    for i in range(50):
        kv.write(keys[i], values[i])
        sizes.append(kv.memory_bytes())
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    assert sizes[-1] == 50 * 32 * 8 * 2


def test_synaptic_memory_is_fixed_size():
    syn = SynapticMemory(dim=32)
    before = syn.memory_bytes()
    keys, values, _ = generate_fact_set(500, 32, seed=1)
    for i in range(500):
        syn.write(keys[i], values[i])
    after = syn.memory_bytes()
    assert before == after == 32 * 32 * 8


def test_kv_cache_perfect_recall_small_set():
    kv = KVCache(dim=64)
    keys, values, labels = generate_fact_set(10, 64, seed=7)
    for k, v in zip(keys, values):
        kv.write(k, v)
    for i in range(10):
        val, sim, _ = kv.retrieve(keys[i])
        assert np.allclose(val, values[i])


def test_synaptic_recall_degrades_with_more_facts():
    small = run_associative_recall(dim=64, num_target_facts=20, num_distractors=30, seed=42)
    large = run_associative_recall(dim=64, num_target_facts=20, num_distractors=5000, seed=42)
    assert small["synaptic_accuracy"] >= large["synaptic_accuracy"]
    assert small["synaptic_memory_bytes"] == large["synaptic_memory_bytes"]
    assert large["kv_memory_bytes"] > small["kv_memory_bytes"]


def test_kv_accuracy_stays_high_regardless_of_length():
    r = run_associative_recall(dim=64, num_target_facts=20, num_distractors=9000, seed=42)
    assert r["kv_accuracy"] >= 0.95


def test_retention_curve_monotonic_decreasing():
    curve = retention_curve(decay=0.95, steps=50)
    r = curve["retention"]
    assert all(r[i] >= r[i + 1] for i in range(len(r) - 1))
    assert r[0] == 1.0


def test_retention_curve_lambda_one_never_decays():
    curve = retention_curve(decay=1.0, steps=50)
    assert all(v == 1.0 for v in curve["retention"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
