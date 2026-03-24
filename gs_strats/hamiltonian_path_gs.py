"""
Ground State Search via Hamiltonian Path Enumeration
=====================================================
DFS-based exact ground state solver for 2D lattice protein folding.

Enumerates self-avoiding walks on a rectangular lattice grid, scores each
path with the Miyazawa-Jernigan (MJ) contact energy, and returns the minimum.
Practical up to ~24-32 residues with a path/time budget cap.

Originally developed in:
    quantum_folding_2d/grid-based/notebooks/benchmark_24res_complete.ipynb

Usage:
    python hamiltonian_path_gs.py
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Amino acid ordering and MJ interaction matrix
# ---------------------------------------------------------------------------

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Miyazawa-Jernigan contact energies (20x20, symmetric, units: kcal/mol)
MJ_MATRIX = np.array([
    [-0.20, -0.34, -0.44, -0.42, -0.62, -0.40, -0.42, -0.29, -0.51, -0.73, -0.65, -0.36, -0.64, -0.69, -0.31, -0.34, -0.40, -0.61, -0.53, -0.68],
    [-0.34, -0.15, -0.44, -0.90, -0.55, -0.50, -0.93, -0.40, -0.33, -0.42, -0.45, -0.35, -0.34, -0.41, -0.24, -0.37, -0.32, -0.26, -0.29, -0.37],
    [-0.44, -0.44, -0.28, -0.46, -0.58, -0.44, -0.51, -0.46, -0.54, -0.51, -0.47, -0.46, -0.41, -0.47, -0.34, -0.48, -0.50, -0.36, -0.42, -0.46],
    [-0.42, -0.90, -0.46, -0.19, -0.52, -0.53, -0.49, -0.44, -0.49, -0.42, -0.35, -0.62, -0.33, -0.36, -0.31, -0.48, -0.44, -0.23, -0.30, -0.36],
    [-0.62, -0.55, -0.58, -0.52, -1.54, -0.65, -0.55, -0.54, -0.73, -0.92, -0.88, -0.47, -0.92, -1.01, -0.49, -0.56, -0.61, -0.84, -0.79, -0.91],
    [-0.40, -0.50, -0.44, -0.53, -0.65, -0.29, -0.55, -0.40, -0.49, -0.56, -0.51, -0.46, -0.47, -0.54, -0.31, -0.43, -0.41, -0.39, -0.42, -0.51],
    [-0.42, -0.93, -0.51, -0.49, -0.55, -0.55, -0.22, -0.43, -0.42, -0.44, -0.37, -0.64, -0.35, -0.40, -0.30, -0.47, -0.43, -0.21, -0.27, -0.38],
    [-0.29, -0.40, -0.46, -0.44, -0.54, -0.40, -0.43, -0.14, -0.46, -0.52, -0.46, -0.41, -0.47, -0.51, -0.26, -0.37, -0.39, -0.44, -0.44, -0.47],
    [-0.51, -0.33, -0.54, -0.49, -0.73, -0.49, -0.42, -0.46, -0.34, -0.66, -0.62, -0.35, -0.64, -0.72, -0.37, -0.46, -0.47, -0.62, -0.57, -0.60],
    [-0.73, -0.42, -0.51, -0.42, -0.92, -0.56, -0.44, -0.52, -0.66, -1.00, -0.95, -0.41, -0.96, -1.04, -0.49, -0.52, -0.59, -0.90, -0.83, -0.98],
    [-0.65, -0.45, -0.47, -0.35, -0.88, -0.51, -0.37, -0.46, -0.62, -0.95, -0.91, -0.37, -0.93, -1.01, -0.46, -0.47, -0.53, -0.89, -0.80, -0.93],
    [-0.36, -0.35, -0.46, -0.62, -0.47, -0.46, -0.64, -0.41, -0.35, -0.41, -0.37, -0.18, -0.32, -0.39, -0.23, -0.39, -0.36, -0.19, -0.24, -0.35],
    [-0.64, -0.34, -0.41, -0.33, -0.92, -0.47, -0.35, -0.47, -0.64, -0.96, -0.93, -0.32, -0.89, -0.98, -0.45, -0.47, -0.52, -0.86, -0.78, -0.92],
    [-0.69, -0.41, -0.47, -0.36, -1.01, -0.54, -0.40, -0.51, -0.72, -1.04, -1.01, -0.39, -0.98, -1.10, -0.50, -0.51, -0.57, -0.98, -0.89, -1.02],
    [-0.31, -0.24, -0.34, -0.31, -0.49, -0.31, -0.30, -0.26, -0.37, -0.49, -0.46, -0.23, -0.45, -0.50, -0.16, -0.30, -0.32, -0.42, -0.38, -0.45],
    [-0.34, -0.37, -0.48, -0.48, -0.56, -0.43, -0.47, -0.37, -0.46, -0.52, -0.47, -0.39, -0.47, -0.51, -0.30, -0.33, -0.41, -0.41, -0.41, -0.48],
    [-0.40, -0.32, -0.50, -0.44, -0.61, -0.41, -0.43, -0.39, -0.47, -0.59, -0.53, -0.36, -0.52, -0.57, -0.32, -0.41, -0.38, -0.45, -0.45, -0.55],
    [-0.61, -0.26, -0.36, -0.23, -0.84, -0.39, -0.21, -0.44, -0.62, -0.90, -0.89, -0.19, -0.86, -0.98, -0.42, -0.41, -0.45, -0.86, -0.76, -0.84],
    [-0.53, -0.29, -0.42, -0.30, -0.79, -0.42, -0.27, -0.44, -0.57, -0.83, -0.80, -0.24, -0.78, -0.89, -0.38, -0.41, -0.45, -0.76, -0.69, -0.78],
    [-0.68, -0.37, -0.46, -0.36, -0.91, -0.51, -0.38, -0.47, -0.60, -0.98, -0.93, -0.35, -0.92, -1.02, -0.45, -0.48, -0.55, -0.84, -0.78, -0.96],
])


# ---------------------------------------------------------------------------
# Lattice and path utilities
# ---------------------------------------------------------------------------

def build_2d_lattice(rows: int, cols: int) -> np.ndarray:
    """Return adjacency matrix for a rows x cols 2D grid lattice."""
    n = rows * cols
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        r, c = i // cols, i % cols
        if c < cols - 1:
            adj[i, i + 1] = adj[i + 1, i] = 1
        if r < rows - 1:
            adj[i, i + cols] = adj[i + cols, i] = 1
    return adj


def find_hamiltonian_paths(
    adj: np.ndarray,
    n: int,
    max_paths: int = 100_000,
    time_limit: float = 300.0,
) -> tuple[list[list[int]], bool]:
    """
    Enumerate self-avoiding walks of length n on the lattice via DFS.

    Args:
        adj: (n x n) adjacency matrix of the lattice.
        n: Number of nodes the path must visit (= number of residues).
        max_paths: Stop after collecting this many paths.
        time_limit: Stop after this many seconds (wall clock).

    Returns:
        paths: List of found paths, each a list of node indices.
        completed: True if enumeration finished without hitting a budget cap.
    """
    paths: list[list[int]] = []
    start_time = time.time()
    completed = True

    def dfs(current: int, visited: set, path: list[int]) -> bool:
        nonlocal completed
        if len(paths) >= max_paths or time.time() - start_time > time_limit:
            completed = False
            return False
        if len(path) == n:
            paths.append(path.copy())
            return True
        for nxt in range(n):
            if nxt not in visited and adj[current, nxt] == 1:
                visited.add(nxt)
                path.append(nxt)
                if not dfs(nxt, visited, path):
                    path.pop()
                    visited.remove(nxt)
                    return False
                path.pop()
                visited.remove(nxt)
        return True

    for start in range(n):
        if len(paths) >= max_paths or time.time() - start_time > time_limit:
            completed = False
            break
        dfs(start, {start}, [start])

    return paths, completed


def compute_path_energy(path: list[int], sequence: str, adj: np.ndarray) -> float:
    """
    Compute the MJ contact energy of a folded sequence along a lattice path.

    Only non-consecutive residue pairs (|i - j| > 1) that land on adjacent
    lattice nodes contribute.
    """
    E = 0.0
    for i in range(len(sequence)):
        for j in range(i + 2, len(sequence)):
            if adj[path[i], path[j]] == 1:
                E += MJ_MATRIX[AA_TO_IDX[sequence[i]], AA_TO_IDX[sequence[j]]]
    return E


def count_contacts(path: list[int], sequence: str, adj: np.ndarray) -> int:
    """Count non-consecutive residue pairs in contact along the path."""
    return sum(
        1
        for i in range(len(sequence))
        for j in range(i + 2, len(sequence))
        if adj[path[i], path[j]] == 1
    )


# ---------------------------------------------------------------------------
# High-level ground state solver
# ---------------------------------------------------------------------------

def find_ground_state(
    sequence: str,
    rows: int,
    cols: int,
    max_paths: int = 100_000,
    time_limit: float = 300.0,
) -> dict:
    """
    Find the ground state energy for a protein sequence on a rows x cols lattice.

    Args:
        sequence: Amino acid sequence (single-letter codes from AMINO_ACIDS).
        rows, cols: Lattice dimensions. rows * cols must be >= len(sequence).
        max_paths: Path enumeration budget.
        time_limit: Wall-clock time budget in seconds.

    Returns:
        dict with keys:
            E_min       - minimum MJ energy found
            best_path   - lattice node indices for the ground state fold
            num_contacts- number of contacts in the ground state
            num_paths   - total paths enumerated
            completed   - whether enumeration was exhaustive
            elapsed     - wall-clock seconds used
    """
    n = len(sequence)
    assert rows * cols >= n, "Lattice too small for sequence"

    adj = build_2d_lattice(rows, cols)
    t0 = time.time()
    paths, completed = find_hamiltonian_paths(adj, n, max_paths=max_paths, time_limit=time_limit)
    elapsed = time.time() - t0

    if not paths:
        return {"E_min": None, "best_path": None, "num_contacts": 0,
                "num_paths": 0, "completed": completed, "elapsed": elapsed}

    scored = [(compute_path_energy(p, sequence, adj), p) for p in paths]
    scored.sort()
    E_min, best_path = scored[0]

    return {
        "E_min": E_min,
        "best_path": best_path,
        "num_contacts": count_contacts(best_path, sequence, adj),
        "num_paths": len(paths),
        "completed": completed,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Small example: 8-residue sequence on a 4x4 lattice
    seq = "ILFVWMCY"
    rows, cols = 4, 4

    print(f"Sequence : {seq}")
    print(f"Lattice  : {rows}x{cols}")
    print(f"Searching...")

    result = find_ground_state(seq, rows, cols, max_paths=50_000, time_limit=60)

    print(f"E_min    : {result['E_min']:.4f} kcal/mol")
    print(f"Contacts : {result['num_contacts']}")
    print(f"Paths    : {result['num_paths']:,} ({'exhaustive' if result['completed'] else 'partial'})")
    print(f"Time     : {result['elapsed']:.2f}s")
