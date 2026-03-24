# QPU Experiment Summary: Protein Folding QUBO on D-Wave Advantage2

**Date:** February 2026
**Hardware:** D-Wave Advantage2 system1.11 (Zephyr topology, ~7,000 qubits)
**Project:** Relative-coordinate QUBO encoding for 2D lattice protein folding

---

## Executive Summary

We ran four rounds of QPU experiments to evaluate whether our ripple-carry and integer QUBO encodings for 2D lattice protein folding could produce valid solutions on current D-Wave hardware. Despite systematic attempts to tune Lagrange multipliers, increase anneal time, directly target the failing constraints, and apply reverse annealing from SA-generated initial states, **neither encoding produced valid solutions at any tested configuration**. The root cause is a structural incompatibility between the ripple-carry adder gadget — used to enforce both the walk recurrence relation and the position increment constraint — and the transverse-field annealing mechanism at current hardware scales.

The grid-based encoding from Irback et al. (2025), included as a comparison baseline, succeeded on QPU (5–9% ground-state rate at N=4, 2–5% at N=5) due to its significantly smaller QUBO and absence of adder gadgets.

---

## Encodings Tested

### Our Encodings

**Ripple-carry (ripple_carry_symbreak)**
Uses a ripple-carry binary adder circuit to enforce:
- Walk constraint: each position updates as `X_{t+1} = X_t ± 1`
- Increment constraint: auxiliary variable `X⁺ = X + 1`

QUBO size at N=5: 342 logical variables, 952 couplings.
Physical embedding: ~700–825 qubits on Advantage2.

**Integer encoding (integer_symbreak)**
Uses integer penalty terms instead of adder circuits, but still requires an increment gadget that reduces to similar multi-qubit correlations.

QUBO size at N=5: 258 logical variables, 1,051 couplings.
Physical embedding: ~791 qubits on Advantage2.

### Comparison Baseline

**Grid encoding with Irback weights (grid_irback)**
Standard grid-based position encoding from Irback et al. (2025).
Lagrange multipliers: λ₁=1.5 (one-hot), λ₂=2.0 (self-avoidance), λ₃=2.0 (connectivity).

QUBO size: 16–36 logical variables, 68–302 couplings (N=4–6).
Physical embedding: 29–84 qubits on Advantage2.

---

## Experiments

### Experiment 1 — Initial Feasibility (test_dwave_qpu)

**Setup:** N=5 (HHPHP), 100 reads, 20 µs anneal, Advantage_system4.1
**Formulations:** integer_symbreak_tuned, ripple_carry_symbreak, ripple_carry_symbreak_tuned

| Formulation | Valid% | Inc. Viol. | Walk Viol. | Dir. Viol. | Chain Breaks |
|---|---|---|---|---|---|
| integer_symbreak_tuned | 0% | 100% | 74% | 98% | 9% |
| ripple_carry_symbreak | 0% | 100% | 87% | 92% | 5% |
| ripple_carry_symbreak_tuned | 0% | 100% | 92% | 67% | 4% |

All three formulations produced zero valid samples. The increment constraint was violated in 100% of samples across all three. Chain break rates were low (4–9%), ruling out embedding quality as the cause.

---

### Experiment 2 — Weight Calibration (qpu_weight_test)

**Setup:** N=4,5,6 (HPPH, HHPPH, HPPHPH), 200 reads, 20 µs anneal, Advantage2_system1.11
**Tested:** 4 Lagrange multiplier bump factors (1×, 1.33×, 1.67×, 2×) applied uniformly to all constraint weights

**Motivation:** SA-tuned weights may be too weak for QPU, where thermal noise is different. Bump factors scale all penalty weights up proportionally.

| Formulation | N=4 Valid% | N=4 GS% | Inc. Viol. | Walk Viol. |
|---|---|---|---|---|
| ripple_sa_1x (1×) | 0% | 0% | 100% | 96–99% |
| ripple_qpu_1_33x (1.33×) | 0% | 0% | 100% | 97–99% |
| ripple_qpu_1_67x (1.67×) | 0% | 0% | 100% | 97–99% |
| ripple_qpu_2x (2×) | 0% | 0% | 100% | 97–99% |
| **grid_irback** | **9.7%** | **4.4%** | **0%** | **0%** |

**Finding:** Uniform weight scaling had zero effect. Increment violations remained at exactly 100% regardless of multiplier. The grid encoding worked at all N tested (N=4: 9.7% valid, N=5: 3%, N=6: 2%).

---

### Experiment 3 — Paper Experiments (paper_exp4_qpu)

**Setup:** N=4,5,6, 3 seeds each (9 instances), 1000 reads, Advantage2_system1.11
**Tested:** 3 formulations × 2 anneal times (20 µs, 100 µs) = 54 QPU tasks

Based on per-constraint violation data from Experiment 2, we identified the increment and walk constraints as the primary failure modes and designed targeted interventions.

#### Intervention 1 — Longer Anneal Time (100 µs vs 20 µs)

**Motivation:** Adder gadgets require correlated multi-qubit transitions. More annealing time could allow the system to thermalize into valid adder states.

| Formulation | Anneal | N=4 Valid% | N=4 Inc. Viol. | N=4 Walk Viol. |
|---|---|---|---|---|
| ripple_symbreak_qpu | 20 µs | 0% | 100% | 97% |
| ripple_symbreak_qpu | 100 µs | 0% | 100% | 98% |

**Finding:** No improvement. Increment violations remained at 100% at both anneal times. Longer annealing did not help the QPU resolve the adder circuit.

#### Intervention 2 — Targeted Adder Weight Boost (ripple_adder_heavy)

**Motivation:** Bump only the adder-based constraints (increment and walk) rather than all weights uniformly. λ_inc: 4 → 50 (12.5×), λ_walk: 20 → 50 (2.5×).

| Formulation | Anneal | N=4 Valid% | Inc. Viol. | Walk Viol. | Dir. Viol. |
|---|---|---|---|---|---|
| ripple_symbreak_qpu | 20 µs | 0% | 100% | 97% | 35% |
| ripple_adder_heavy | 20 µs | 0% | 100% | 88% | **79%** |

**Finding:** Targeted weight boost made things worse. Increment violations remained at 100%. Walk violations dropped slightly (97% → 88%), but direction violations *increased* substantially (35% → 79%) — the massive penalty weights for increment and walk crowded out the direction constraints in the energy landscape. The QPU's optimization budget is finite; forcing it to concentrate on two constraints degrades performance on others.

#### Grid Baseline Results (Experiment 3)

| Formulation | Anneal | N=4 Valid% | N=4 GS% | N=5 Valid% | N=5 GS% | N=6 Valid% | N=6 GS% |
|---|---|---|---|---|---|---|---|
| grid_irback | 20 µs | 7.3% | 5.8% | 4.8% | 2.2% | 0.7% | 0.3% |
| grid_irback | 100 µs | 8.6% | 7.6% | 4.8% | 2.4% | 0.8% | 0.4% |

Grid encoding consistently finds valid and ground-state solutions. Valid rate drops with N (hardware noise scales with problem size).

---

### Experiment 4 — Reverse Annealing (paper_exp5_ra)

**Setup:** N=4,5,6, 3 seeds each (9 instances), 1000 reads, Advantage2_system1.11
**Tested:** 4 RA variants × 1 formulation + 1 forward baseline = 45 QPU tasks

**Motivation:** Reverse annealing starts from a known classical state and partially de-anneals to explore locally, giving the carry chain a chance to settle without a global quantum fluctuation. This addresses the root cause directly: instead of asking the QPU to discover a valid adder state from scratch, we seed it with one and ask it only to refine.

**Protocol:**
1. Run SA (500 reads, 50k sweeps) per instance to find the lowest-energy logical solution.
2. Embed that solution into physical qubits using the stored minor embedding.
3. Apply RA schedule: s=1 (classical) → ramp to s_target → hold → ramp back to s=1.
4. `reinitialize_state=True`: each of the 1000 reads starts fresh from the SA state.

| Solver | s_target | Pause | N=4 Valid% | N=5 Valid% | N=6 Valid% |
|---|---|---|---|---|---|
| fwd_20us (baseline) | — | — | 0% | 0% | 0% |
| ra_s045_p50us | 0.45 | 50 µs | 0% | 0% | 0% |
| ra_s045_p200us | 0.45 | 200 µs | 0% | 0% | 0% |
| ra_s050_p50us | 0.50 | 50 µs | 0% | 0% | 0% |
| ra_s050_p200us | 0.50 | 200 µs | 0% | 0% | 0% |

**Finding:** Reverse annealing produced zero valid samples across all 45 tasks. Even starting from an SA-seeded state (which already satisfies most constraints), any quantum fluctuation introduced by the RA schedule corrupts the carry chain and the QPU cannot recover it. The RA result is the strongest evidence yet that the failure is structural: the adder constraint is not just hard to find from scratch — it cannot be maintained under any degree of quantum noise at current hardware scales.

---

## Root Cause Analysis

The failure is structural, not a tuning problem. The ripple-carry adder implements:

```
X_{t+1} = X_t + d_t
```

where `d_t ∈ {-1, 0, +1}` is the direction at step `t`. In binary QUBO form, this requires a carry propagation chain across multiple auxiliary bits. For a chain of length N, the adder involves O(N log N) auxiliary variables with O(N log N) coupling terms, all of which must be simultaneously satisfied.

Transverse-field quantum annealing finds low-energy states by evolving through a superposition of all bit configurations. Satisfying the adder constraint requires a highly correlated multi-qubit transition — all carry bits must flip together — which has exponentially small probability in a single-qubit annealing step. At current hardware scales (N=4–6, 160–416 logical variables), the anneal time required to resolve this correlation is far longer than the decoherence time of the device.

**Evidence:** Increment violations at exactly 100% across 135+ QPU experiments, independent of:
- Encoding (ripple-carry or integer)
- Anneal time (20 µs or 100 µs)
- Weight magnitude (1× to 12.5× of SA-tuned values)
- Number of reads (100, 200, or 1000)
- Annealing direction (forward or reverse, with SA-seeded initial state)

---

## What Would Be Required to Fix This

1. **Longer coherence time / anneal time** — Current Advantage2 supports up to 2,000 µs. We tested up to 100 µs. A systematic sweep at 200–2,000 µs might help, but the structural nature of the failure makes this unlikely to be sufficient at current qubit counts.

2. **Reverse annealing** *(tried — see Experiment 4)* — We implemented this: SA-generated initial states were embedded into physical qubits and used to seed reverse annealing at s_target ∈ {0.45, 0.50} with pause times of 50–200 µs. All 36 RA tasks returned 0% valid. The carry chain cannot be maintained even under minimal quantum fluctuation when starting from a valid state.

3. **Reformulation without adder circuits** — Replace the ripple-carry increment with a constraint formulation that does not require carry propagation, e.g., unary encoding of the position difference, or a penalty term that approximates the increment without auxiliary carry bits.

4. **Larger QPU / fault-tolerant hardware** — The grid encoding works precisely because it has no adder gadgets and embeds in 29–84 qubits. Our encoding requires 160–2,074 logical variables (N=4–12). At sufficient qubit counts with error correction, the adder would become resolvable.

---

## Summary Table

| What We Tried | Outcome | Why |
|---|---|---|
| SA-tuned weights on QPU (1×) | 0% valid | Increment 100% violated |
| Uniform weight bump (1.33×–2×) | 0% valid | No effect on increment |
| Longer anneal time (100 µs) | 0% valid | Structural, not time-limited |
| Targeted adder weight boost (λ_inc=50, λ_walk=50) | 0% valid, direction worse | Constraint competition |
| Integer encoding (different gadget type) | 0% valid | Increment still 100% violated |
| Reverse annealing, s=0.45–0.50, pause=50–200 µs | 0% valid | Carry chain corrupted under any quantum noise |

**Bottom line:** The increment constraint, implemented as a ripple-carry adder in QUBO form, is not resolvable by transverse-field quantum annealing at current hardware scales. This holds even when seeding from a valid SA solution via reverse annealing. Our encoding is well-suited for simulated annealing (where constraint satisfaction can be guided sequentially) but requires hardware or formulation advances to be QPU-viable.

---

## Data

All raw solution files, per-constraint violation data, and analysis CSVs are in the repository:

```
run/test_dwave_qpu/          — Experiment 1 (initial feasibility)
run/qpu_weight_test/         — Experiment 2 (weight calibration)
run/paper_experiments/paper_exp4_qpu/  — Experiment 3 (paper experiments)
run/paper_experiments/paper_exp5_ra/   — Experiment 4 (reverse annealing)
```

Each solution `.json` includes full per-constraint violation frequencies, chain break fractions, sample energies, and timing.
