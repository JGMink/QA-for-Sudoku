# QPU Experiment Proposal: MJ Protein Folding on D-Wave Advantage2

**To:** Prof. Dinh
**From:** Jonah Minkoff
**Date:** March 2026

---

Two runs on Sycamore (D-Wave Advantage2 system1.11, Zephyr, ~7,000 qubits) to get
QPU valid/ground-state rates and chain break fractions across problem sizes for the
poster, complementing the existing SA benchmarks.

## What

**Run 1 — Weight validation (8-residue)**
The SA-optimal penalty weights (λ₁=3, λ₂=4, λ₃=4) may not be optimal on QPU.
Run 3 weight configurations across all 12 test sequences to identify the best config
before committing to the scaling study.

| Config | λ₁ | λ₂ | λ₃ |
|---|---|---|---|
| SA-direct | 3.0 | 4.0 | 4.0 |
| QPU-1.33× | 4.0 | 5.3 | 5.3 |
| Irback-style | 1.5 | 2.0 | 2.0 |

36 tasks × 1,000 reads × 20 µs → **~12.6 s QPU time**

**Run 2 — Scaling study (8 / 12 / 16-residue)**
Using the best config from Run 1, test 4 sequences per problem size. This is the
poster figure: QPU valid/ground-state rate vs. problem size, alongside the existing
SA baseline.

Metrics: valid rate, ground-state rate, chain break fraction per size.

12 tasks × 1,000 reads × 20 µs → **~4.2 s QPU time**

**Total: 48 tasks, ~17 s QPU access time**

## Timeline

| Step | Where | Time |
|---|---|---|
| Build pipeline + extend QUBO to 12/16-res | Local | ~1 day |
| Phases 1–2: QUBO generation + minor embedding | Sycamore | ~1–2 hrs |
| Phase 3: QPU solve (48 tasks, ~10–20 s API round-trip each) | Sycamore | ~15–20 min |
| Phase 4: analysis + poster figures | Local | ~2 hrs |

**Start to poster-ready figures: 2 days.** Bottleneck is day 1 (pipeline build).
Sycamore phases run largely hands-off.
