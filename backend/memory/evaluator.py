"""
Evaluator
---------
Controlled associative-recall benchmark used to compare KV Cache vs
Synaptic Memory honestly. Everything here is computed, not hardcoded.

Two vector sources are used:
1. `random_unit_vectors` — pure random seeded vectors for the abstract
   "person_001 -> label" benchmark (Benchmark Lab / Playground charts).
2. `text_to_vector` — a DISCLOSED, deterministic hash-seeded pseudo-embedding
   used only so free-text demo facts like "Aditya likes basketball" can be
   turned into vectors for the Recall Lab demo. This is NOT a real language
   embedding model — it has no semantic understanding, it only guarantees
   the same string always maps to the same vector and different strings map
   to (with high probability) different vectors. This limitation is stated
   explicitly in the README and UI.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np

from .kv_cache import KVCache
from .synaptic_memory import SynapticMemory


def random_unit_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


def text_to_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic hash-seeded pseudo-embedding. Disclosed toy embedding — NOT a real LM embedding."""
    h = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / (np.linalg.norm(v) + 1e-9)


def generate_fact_set(num_facts: int, dim: int, seed: int):
    """Controlled key/value vocabulary: person_i -> label_i."""
    keys = random_unit_vectors(num_facts, dim, seed)
    values = random_unit_vectors(num_facts, dim, seed + 1)
    labels = [f"label_{i:05d}" for i in range(num_facts)]
    return keys, values, labels


def run_associative_recall(
    dim: int,
    num_target_facts: int,
    num_distractors: int,
    seed: int = 42,
    decay: float = 0.99,
    lr: float = 1.0,
):
    """
    Writes num_target_facts "important" facts interleaved randomly among
    num_distractors distractor facts into both a KV cache and a synaptic
    memory, in the SAME order for both, then queries the target facts and
    measures whether each system retrieves the correct value.

    Returns a dict of real, computed metrics (no fabricated numbers).
    """
    total_facts = num_target_facts + num_distractors
    keys, values, labels = generate_fact_set(total_facts, dim, seed)

    # target facts are the first num_target_facts; then shuffle write order
    # so target facts are scattered across the sequence, not all at the end.
    order = np.arange(total_facts)
    rng = np.random.default_rng(seed + 2)
    rng.shuffle(order)

    kv = KVCache(dim=dim)
    syn = SynapticMemory(dim=dim, decay=decay, lr=lr)

    write_position = {}  # fact_idx -> position in write order
    for pos, idx in enumerate(order):
        kv.write(keys[idx], values[idx])
        syn.write(keys[idx], values[idx])
        write_position[int(idx)] = pos

    target_idxs = list(range(num_target_facts))

    kv_correct, syn_correct = 0, 0
    kv_lat_total, syn_lat_total = 0.0, 0.0
    per_fact = []

    for idx in target_idxs:
        query = keys[idx]

        t0 = time.perf_counter()
        kv_val, kv_sim, _ = kv.retrieve(query)
        kv_lat_total += time.perf_counter() - t0
        kv_pred_label, kv_pred_sim = _decode(kv_val, values, labels)
        kv_ok = kv_pred_label == labels[idx]
        kv_correct += int(kv_ok)

        t0 = time.perf_counter()
        syn_label, syn_sim, _raw = syn.retrieve_from_candidates(query, values, labels)
        syn_lat_total += time.perf_counter() - t0
        syn_ok = syn_label == labels[idx]
        syn_correct += int(syn_ok)

        per_fact.append({
            "fact_index": idx,
            "write_position": write_position[idx],
            "kv_correct": kv_ok,
            "synaptic_correct": syn_ok,
            "synaptic_similarity": syn_sim,
        })

    n = max(num_target_facts, 1)

    # Forgetting/interference signal: accuracy for facts written EARLY
    # (first half of the sequence) vs LATE (second half), synaptic only.
    early = [p for p in per_fact if p["write_position"] < total_facts / 2]
    late = [p for p in per_fact if p["write_position"] >= total_facts / 2]
    syn_acc_early = (sum(p["synaptic_correct"] for p in early) / len(early)) if early else None
    syn_acc_late = (sum(p["synaptic_correct"] for p in late) / len(late)) if late else None

    return {
        "sequence_length": total_facts,
        "num_target_facts": num_target_facts,
        "num_distractors": num_distractors,
        "dim": dim,
        "decay": decay,
        "lr": lr,
        "kv_accuracy": kv_correct / n,
        "synaptic_accuracy": syn_correct / n,
        "kv_memory_bytes": kv.memory_bytes(),
        "synaptic_memory_bytes": syn.memory_bytes(),
        "kv_latency_ms_per_query": (kv_lat_total / n) * 1000,
        "synaptic_latency_ms_per_query": (syn_lat_total / n) * 1000,
        "synaptic_accuracy_early_writes": syn_acc_early,
        "synaptic_accuracy_late_writes": syn_acc_late,
        "interference_gap": (
            (syn_acc_late - syn_acc_early) if (syn_acc_early is not None and syn_acc_late is not None) else None
        ),
        "per_fact": per_fact,
    }


def _decode(raw_value: np.ndarray, candidate_values: np.ndarray, labels: list[str]):
    norm_raw = raw_value / (np.linalg.norm(raw_value) + 1e-9)
    Cn = candidate_values / (np.linalg.norm(candidate_values, axis=1, keepdims=True) + 1e-9)
    sims = Cn @ norm_raw
    best_idx = int(np.argmax(sims))
    return labels[best_idx], float(sims[best_idx])


def run_benchmark(
    sequence_lengths: list[int],
    dim: int = 64,
    num_target_facts: int = 20,
    seed: int = 42,
    decay: float = 0.99,
    lr: float = 1.0,
):
    """Runs run_associative_recall across multiple sequence lengths. Real computation, every point."""
    results = []
    for seq_len in sequence_lengths:
        distractors = max(seq_len - num_target_facts, 0)
        r = run_associative_recall(
            dim=dim,
            num_target_facts=num_target_facts,
            num_distractors=distractors,
            seed=seed,
            decay=decay,
            lr=lr,
        )
        r.pop("per_fact")  # keep benchmark payload light
        results.append(r)
    return results


def retention_curve(decay: float, steps: int = 200):
    """
    Theoretical retention curve for a single written memory under repeated
    decay with no further writes to that memory: strength_t = decay ** t.
    This is the exact, real closed-form of the update rule's decay term
    (eta contribution from other writes ignored to isolate lambda's effect).
    """
    t = np.arange(steps)
    strength = decay ** t
    return {"t": t.tolist(), "retention": strength.tolist()}
