"""
Run this to regenerate results/benchmark_results.json from scratch.
Every number is computed live — nothing here is hand-typed.

Usage:
    python experiments/benchmark.py
"""
import sys
import os
import json
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from memory.evaluator import run_benchmark, retention_curve  # noqa: E402


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    sequence_lengths = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000]

    configs = [
        {"name": "default", "dim": 64, "decay": 0.99, "lr": 1.0, "num_target_facts": 20},
        {"name": "high_decay", "dim": 64, "decay": 0.999, "lr": 1.0, "num_target_facts": 20},
        {"name": "low_decay", "dim": 64, "decay": 0.9, "lr": 1.0, "num_target_facts": 20},
        {"name": "larger_memory", "dim": 256, "decay": 0.99, "lr": 1.0, "num_target_facts": 20},
    ]

    output = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": "All values below are computed directly by backend/memory/evaluator.py. "
                "Reproduce with: python experiments/benchmark.py",
        "sweeps": {},
        "retention_curves": {},
    }

    for cfg in configs:
        print(f"Running sweep: {cfg['name']} ...")
        results = run_benchmark(
            sequence_lengths=sequence_lengths,
            dim=cfg["dim"],
            num_target_facts=cfg["num_target_facts"],
            seed=42,
            decay=cfg["decay"],
            lr=cfg["lr"],
        )
        output["sweeps"][cfg["name"]] = {"config": cfg, "results": results}

    for decay in [0.90, 0.95, 0.99, 0.999, 1.0]:
        output["retention_curves"][str(decay)] = retention_curve(decay=decay, steps=200)

    out_path = os.path.join(out_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")

    # Print a quick human-readable summary
    default = output["sweeps"]["default"]["results"]
    print("\n--- default config summary ---")
    for r in default:
        print(
            f"seq={r['sequence_length']:>6}  "
            f"KV_acc={r['kv_accuracy']:.2f}  Syn_acc={r['synaptic_accuracy']:.2f}  "
            f"KV_mem={r['kv_memory_bytes']:>9}B  Syn_mem={r['synaptic_memory_bytes']:>7}B"
        )


if __name__ == "__main__":
    main()
