"""
Precompute ground-state energies for all QPU experiment sequences.

Uses Hamiltonian path enumeration (exact) from gs_strats/hamiltonian_path_gs.py.
Results are cached to qpu_experiments/ground_truths.json and consumed by run_qpu.py.

Usage (run once, from repo root):
    python qpu_experiments/compute_ground_truths.py

Notes:
    8-res  (2x4 lattice): enumeration completes in seconds — exact.
    12-res (3x4 lattice): enumeration completes in ~1-5 min — exact.
    16-res (4x4 lattice): may hit max_paths budget — lower bound, flagged as partial.
"""

import json
import sys
import time
from pathlib import Path

# Allow imports from repo root (gs_strats lives there)
sys.path.insert(0, str(Path(__file__).parent.parent))

from gs_strats.hamiltonian_path_gs import find_ground_state  # noqa: E402

# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

# Run 1: all 12 eight-residue sequences (from benchmark_8res_complete notebook)
SEQ_8 = [
    "CFIKDLWV", "ILFVWMCY", "CFWVILMY", "WFYCMILV",
    "ACDEFGHI", "KLMNPQRS", "STVWYACF", "RKDECFWI",
    "RKDENQST", "HHKKRRDD", "SSTTNNQQ", "GPAGPSAG",
]

# Run 2 subsets (4 per size, selected for diversity)
SEQ_8_RUN2  = ["CFIKDLWV", "ILFVWMCY", "WFYCMILV", "RKDECFWI"]

SEQ_12_RUN2 = [
    "CFIKDLWVRKDE",   # hydrophobic core + charged tail
    "ILFVWMCYHHKK",   # hydrophobic + basic
    "WFYCMILVSSTT",   # aromatic + polar
    "ACDEFGHIKLMN",   # alphabetically diverse
]

SEQ_16_RUN2 = [
    "CFIKDLWVRKDENQST",   # 16-res diverse
    "ILFVWMCYHHKKRRDD",   # hydrophobic + charged
    "WFYCMILVSSTTNNQQ",   # aromatic + polar
    "ACDEFGHIKLMNPQRS",   # alphabetically diverse
]

# Lattice configs per problem size
LATTICES = {8: (2, 4), 12: (3, 4), 16: (4, 4)}

# Budget caps (exact for 8/12, partial lower-bound OK for 16)
BUDGETS = {
    8:  {"max_paths": 500_000, "time_limit": 120.0},
    12: {"max_paths": 500_000, "time_limit": 300.0},
    16: {"max_paths": 500_000, "time_limit": 300.0},
}


def compute_all(out_path: Path) -> None:
    results = {}

    all_tasks = [
        (8,  SEQ_8),
        (8,  SEQ_8_RUN2),
        (12, SEQ_12_RUN2),
        (16, SEQ_16_RUN2),
    ]

    # Deduplicate while preserving order
    seen = set()
    tasks: list[tuple[int, str]] = []
    for size, seqs in all_tasks:
        for seq in seqs:
            key = (size, seq)
            if key not in seen:
                seen.add(key)
                tasks.append(key)

    print(f"Computing ground truths for {len(tasks)} (size, sequence) pairs\n")

    for i, (size, seq) in enumerate(tasks):
        rows, cols = LATTICES[size]
        budget = BUDGETS[size]
        print(f"[{i+1}/{len(tasks)}] {seq}  ({rows}x{cols})", flush=True)

        t0 = time.time()
        gs = find_ground_state(
            seq, rows, cols,
            max_paths=budget["max_paths"],
            time_limit=budget["time_limit"],
        )
        elapsed = time.time() - t0

        tag = "exact" if gs["completed"] else "partial"
        print(f"         E_min={gs['E_min']:.4f}  paths={gs['num_paths']:,}  "
              f"contacts={gs['num_contacts']}  [{tag}]  {elapsed:.1f}s", flush=True)

        results[f"{size}_{seq}"] = {
            "sequence":    seq,
            "n_residues":  size,
            "lattice":     f"{rows}x{cols}",
            "E_min":       gs["E_min"],
            "best_path":   gs["best_path"],
            "num_contacts": gs["num_contacts"],
            "num_paths":   gs["num_paths"],
            "completed":   gs["completed"],
            "elapsed_s":   round(elapsed, 2),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} entries -> {out_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "ground_truths.json"
    if out.exists():
        print(f"Already exists: {out}")
        print("Delete it to recompute, or it will be loaded directly by run_qpu.py.")
    else:
        compute_all(out)
