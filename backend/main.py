"""
Memory Frontier — Backend API
------------------------------
Every number this API returns comes from an actual computation on
request (or, for the benchmark sweep, from a request-time run across
multiple sequence lengths). Nothing is hardcoded.
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memory.evaluator import (
    run_associative_recall,
    run_benchmark,
    retention_curve,
    text_to_vector,
    generate_fact_set,
)
from memory.synaptic_memory import SynapticMemory
from memory.kv_cache import KVCache

app = FastAPI(title="Memory Frontier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for hackathon demo; tighten if needed post-submission
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class ExperimentRequest(BaseModel):
    dim: int = Field(64, ge=8, le=512)
    num_target_facts: int = Field(20, ge=1, le=200)
    num_distractors: int = Field(200, ge=0, le=50000)
    decay: float = Field(0.99, gt=0, le=1.0)
    lr: float = Field(1.0, gt=0, le=10.0)
    seed: int = 42


class BenchmarkRequest(BaseModel):
    sequence_lengths: list[int] = Field(default=[50, 200, 500, 1000, 3000, 6000, 10000])
    dim: int = Field(64, ge=8, le=512)
    num_target_facts: int = Field(20, ge=1, le=200)
    decay: float = Field(0.99, gt=0, le=1.0)
    lr: float = Field(1.0, gt=0, le=10.0)
    seed: int = 42


class RetentionRequest(BaseModel):
    decay: float = Field(0.99, gt=0, le=1.0)
    steps: int = Field(200, ge=1, le=5000)


class Fact(BaseModel):
    text: str
    query_key_hint: Optional[str] = None  # e.g. "Aditya" -- the subject to query on


class RecallTestRequest(BaseModel):
    facts: list[str]
    query: str
    num_distractors: int = Field(200, ge=0, le=20000)
    dim: int = Field(64, ge=8, le=512)
    decay: float = Field(0.99, gt=0, le=1.0)
    lr: float = Field(1.0, gt=0, le=10.0)
    seed: int = 42


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/experiment/run")
def experiment_run(req: ExperimentRequest):
    result = run_associative_recall(
        dim=req.dim,
        num_target_facts=req.num_target_facts,
        num_distractors=req.num_distractors,
        seed=req.seed,
        decay=req.decay,
        lr=req.lr,
    )
    return result


@app.post("/benchmark/run")
def benchmark_run(req: BenchmarkRequest):
    if any(sl < req.num_target_facts for sl in req.sequence_lengths):
        raise HTTPException(400, "sequence_length must be >= num_target_facts")
    results = run_benchmark(
        sequence_lengths=req.sequence_lengths,
        dim=req.dim,
        num_target_facts=req.num_target_facts,
        seed=req.seed,
        decay=req.decay,
        lr=req.lr,
    )
    return {"results": results}


@app.post("/memory/retention")
def memory_retention(req: RetentionRequest):
    return retention_curve(decay=req.decay, steps=req.steps)


@app.post("/recall/test")
def recall_test(req: RecallTestRequest):
    """
    Free-text recall demo. Facts are turned into vectors with a disclosed
    deterministic hash-based pseudo-embedding (NOT a real language model
    embedding — see README/UI note). Distractors are random controlled
    vectors added to the same memories to simulate a long context.
    """
    if not req.facts:
        raise HTTPException(400, "Provide at least one fact")

    fact_keys = np.stack([text_to_vector(f, req.dim) for f in req.facts])
    fact_values = np.stack([text_to_vector(f + "::VALUE", req.dim) for f in req.facts])
    fact_labels = req.facts

    distractor_keys, distractor_values, distractor_labels = generate_fact_set(
        req.num_distractors, req.dim, seed=req.seed
    )

    all_keys = np.vstack([fact_keys, distractor_keys]) if req.num_distractors > 0 else fact_keys
    all_values = np.vstack([fact_values, distractor_values]) if req.num_distractors > 0 else fact_values
    all_labels = fact_labels + distractor_labels

    order = np.arange(len(all_labels))
    rng = np.random.default_rng(req.seed + 3)
    rng.shuffle(order)

    kv = KVCache(dim=req.dim)
    syn = SynapticMemory(dim=req.dim, decay=req.decay, lr=req.lr)
    for idx in order:
        kv.write(all_keys[idx], all_values[idx])
        syn.write(all_keys[idx], all_values[idx])

    query_vec = text_to_vector(req.query, req.dim)

    kv_val, kv_sim, _ = kv.retrieve(query_vec)
    kv_label, kv_label_sim = _decode_label(kv_val, all_values, all_labels)

    syn_label, syn_sim, _raw = syn.retrieve_from_candidates(query_vec, all_values, all_labels)

    return {
        "query": req.query,
        "sequence_length": len(all_labels),
        "kv_result": {"retrieved_fact": kv_label, "similarity": kv_label_sim},
        "synaptic_result": {"retrieved_fact": syn_label, "similarity": syn_sim},
        "kv_memory_bytes": kv.memory_bytes(),
        "synaptic_memory_bytes": syn.memory_bytes(),
        "note": "Facts are embedded with a deterministic hash-based toy embedding, not a real language model.",
    }


def _decode_label(raw_value, candidate_values, labels):
    norm_raw = raw_value / (np.linalg.norm(raw_value) + 1e-9)
    Cn = candidate_values / (np.linalg.norm(candidate_values, axis=1, keepdims=True) + 1e-9)
    sims = Cn @ norm_raw
    best_idx = int(np.argmax(sims))
    return labels[best_idx], float(sims[best_idx])


@app.get("/experiment/demo")
def judge_demo():
    """
    One-click 'Judge Demo': a short, reproducible experiment answering the
    core research question in under a few seconds.
    """
    bench = run_benchmark(
        sequence_lengths=[50, 500, 2000, 5000, 10000],
        dim=64,
        num_target_facts=20,
        seed=42,
        decay=0.99,
        lr=1.0,
    )
    return {
        "claim": (
            "A fixed-size Hebbian synaptic memory can match KV-cache retrieval "
            "accuracy at small context sizes, but accuracy degrades through "
            "interference as more facts are written, while its memory footprint "
            "stays constant — unlike the KV cache, whose memory grows linearly "
            "but whose accuracy stays near-perfect."
        ),
        "results": bench,
    }


# ---------- Serve the frontend (static site) from the SAME Railway service ----------
# This must be mounted LAST so it does not shadow the API routes above.
# Frontend lives at ../frontend relative to this file, i.e. project_root/frontend.
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
