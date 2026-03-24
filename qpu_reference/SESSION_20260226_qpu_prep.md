# Session Summary: QPU Tuning Analysis & Config Preparation
**Date**: 2026-02-25 to 2026-02-26
**Duration**: Multi-session (context limit reached once, continued)
**Status**: ✅ Complete — waiting for SLURM job 14224 before executing QPU runs

---

## Session Goal

Prepare to run all QPU-based paper experiments (Experiment 4, pure QPU / no hybrid solver).
Before running, needed to: analyze all QPU trial data, fix the broken paper_exp4_qpu config,
and create QPU-specific tuned weights.

---

## What We Did

### 1. Monitored SLURM Job 14224 (generalized_tuning)

Job was running on Sycamore at ~40 Python workers, 90-100% CPU.

**Problem found**: `results/generalized_tuning_20260225_134657/` never created, so all
checkpoints are silently failing. This is because `_save_checkpoint()` wraps the file write
in a try/except that catches FileNotFoundError, and the output directory was never `mkdir`'d.

**Also**: The sbatch script was modified to add `--skip-ratio --eval-sweeps-cap 200000` AFTER
submission. SLURM captured the original script, so the job is running the full ratio sweep.

**Decision**: No way to safely extract partial results (all data in-memory in workers).
Wait for job to finish naturally (~4-5 more hours from when we checked).

**Command to pull results when done**:
```bash
scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/results/generalized_tuning_*/ \
    C:\Users\jonah\Documents\Coding\School\quantum_folding_2d\results\
```

### 2. Analyzed All QPU Trial Data

**Files read**:
- `run/test_dwave_qpu/` — N=5, 3 encodings, 100 reads, Advantage_system4.1
- `run/run_test_dwave/` — N=4-5, 4 formulations (int_4/rc_4/int_15/rc_15), 500 reads
- `run/test_strong_penalties/` — N=4, integer λ=15, 500 reads, Advantage_system4.1
- `results/focal_tuning_20260210_174046/` — 8,225 rows SA tuning data

**Critical findings**:

#### Ripple >>> Integer on QPU
| Encoding | N=4 QPU energy | N=5 QPU energy |
|---|---|---|
| int_optimal_4 | 191.8 | 918.3 |
| ripple_carry_4 | **140.0** | **488.1** |
| ripple_carry_symbreak_tuned | — | **833.7** |
| integer_symbreak_tuned | — | 1111.4 |

Ripple wins by 27-47% on QPU at same weight scale. Ripple has fewer quads (812 vs 1051 at N=5),
leading to lower chain strength and better embedding.

#### Chain Break Data (the most important QPU safety metric)
| Config | λ_dir | chain_strength | chain_breaks |
|---|---|---|---|
| ripple_carry_symbreak_tuned | 30 | 132.7 | **4%** ✅ |
| integer_symbreak_tuned | 15 | 308.8 | 9% ✅ |
| integer at λ=15 (test_strong_penalties) | 15 | 372.7 | **22.8%** ⚠️ DANGER |

**Key insight**: Integer encoding at λ=15 hits the >20% chain break danger zone. Integer CANNOT
be bumped hard. Ripple at λ=30 has wide safe headroom for a moderate bump.

#### Physical embedding stats (N=5, from test_dwave_qpu solutions)
| Encoding | Physical qubits | Max chain |
|---|---|---|
| integer_symbreak_tuned | 791 | 12 |
| ripple_carry_symbreak | 688 | 11 |
| ripple_carry_symbreak_tuned | 800 | **15** |

Note: ripple_tuned has more physical qubits than ripple_untuned at N=5. The higher λ values
lead to a denser effective coupling and slightly larger embedding. Still acceptable.

### 3. Designed QPU Weight Strategy

The SA-tuned presets are good for SA but suboptimal for QPU because:
1. λ_adj=3 is relatively high → adjacency gadgets add significant chain strain
2. Constraints at λ=30 are adequate for SA validity but QPU can benefit from slightly harder walls

**QPU strategy**: 1.33× constraint bump + reduce λ_adj from 3 → 1.

This:
- Hardens validity energy landscape (fewer constraint violations on QPU)
- Widens gap between valid/invalid states relative to GS energy signal
- Keeps chain_strength projected at ~177 (vs danger zone at ~370)

### 4. Added `ripple_carry_symbreak_qpu` Preset

**File**: `src/constraint_catalog.py`

```python
'ripple_carry_symbreak_qpu': [
    ConstraintSpec('direction', build_direction_one_hot, 'lambda_dir', 40.0),
    ConstraintSpec('symmetry', build_symmetry_breaking, 'lambda_sym', 40.0),
    ConstraintSpec('walk', build_walk_ripple_carry_v3, 'lambda_walk', 20.0),
    ConstraintSpec('parity', build_parity_constraint, 'lambda_parity', 20.0),
    ConstraintSpec('increment', build_increment_ripple_carry, 'lambda_inc', 4.0),
    ConstraintSpec('collision', build_collision_eq_gadget, 'lambda_coll', 4.0),
    ConstraintSpec('objective', build_mj_objective, 'lambda_adj', 1.0),
],
```

### 5. Fixed `run/paper_experiments/paper_exp4_qpu/config.yaml`

**Changes made**:

| Field | Before | After | Reason |
|---|---|---|---|
| `solver_id` | `Advantage_system6.4` | `Advantage2_system1.11` | 6.4 not available; 2.x is Zephyr (HPC guide says use 2.x for paper) |
| `formulation` | `hybrid_ripple` (ripple_carry_symbreak, λ_dir=25 untuned) | `ripple_symbreak_qpu` (new QPU preset) | Use QPU-specific weights, no manual overrides needed |
| `timeout` | 120s | 600s | More reliable embeddings for paper results |
| `tries` | 20 | 50 | Fewer embedding failures |
| `max_no_improvement` | 20 | 50 | Match tries |

Kept: num_reads=1000 ✅, annealing_time=20µs ✅ (standard benchmark value),
chain_strength=null ✅ (auto), brute_force=true ✅ (N=4-6 all fast)

---

## Files Changed

| File | Change |
|---|---|
| `src/constraint_catalog.py` | Added `ripple_carry_symbreak_qpu` preset (~20 lines) |
| `run/paper_experiments/paper_exp4_qpu/config.yaml` | Complete rewrite with fixes |
| `src/CLAUDE_GUIDE.md` | Updated PRESETS table, added QPU chain break threshold table, clarified λ_adj |
| `run/CLAUDE_GUIDE.md` | Updated run status table, updated tuned weight table with QPU preset note |
| `paper_experiments/CLAUDE_GUIDE.md` | Updated Group 5 section with full config details and trial data |
| `timeline/TIMELINE.md` | Added Feb 26 session entry |
| `timeline/SESSION_20260226_qpu_prep.md` | This file |

---

## What to Do When Continuing This Work

### Step 1: Pull gen_tuning results

```bash
# On local machine
scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/results/generalized_tuning_20260225_134657/ \
    C:\Users\jonah\Documents\Coding\School\quantum_folding_2d\results\
```

### Step 2: Analyze generalized_tuning output

Look at `results/generalized_tuning_20260225_134657/phase_summaries.json` and the final CSV.
Key question: do the recommended weight ratios for ripple-carry match our QPU preset?
If the ratio recommendations suggest λ_dir/λ_inc should be closer to 15× (vs our 10×), consider
bumping λ_dir slightly (e.g., 40→50) while keeping λ_adj=1.

### Step 3: Run paper_exp4_qpu on Sycamore

```bash
ssh minkoffjg@sycamore.cs.vcu.edu
cd ~/quantum_folding_2d
git pull  # get the new preset + config

source venv/bin/activate

# Run phases 1+2 first (free, no QPU)
python -m benchmarks.pipeline.runner run/paper_experiments/paper_exp4_qpu --phases 1,2

# Verify: should see 9 instances (N=4,5,6 × seeds 1,2,3)
ls run/paper_experiments/paper_exp4_qpu/instances/

# Run phase 3 (QPU — consumes Leap credits, ~0.18s total)
nohup python -m benchmarks.pipeline.runner run/paper_experiments/paper_exp4_qpu --phases 3 \
    > run/paper_experiments/paper_exp4_qpu/run.log 2>&1 &
tail -f run/paper_experiments/paper_exp4_qpu/run.log

# Run phase 4 (free)
python -m benchmarks.pipeline.runner run/paper_experiments/paper_exp4_qpu --phases 4
```

### Step 4: Pull results

```bash
scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/run/paper_experiments/paper_exp4_qpu/ \
    C:\Users\jonah\Documents\Coding\School\quantum_folding_2d\run\paper_experiments\
```

### Step 5: Evaluate decision gate

Check `run/paper_experiments/paper_exp4_qpu/analysis/metrics_summary.csv`:
- **Pass**: max_chain ≤ 15 at N=6 AND chain_breaks ≤ 10% → add grid encoding, proceed to Phase 2 (N=8,10,12)
- **Fail**: Revise QPU narrative, reduce scope to N=4-6 comparison only

---

## Key Non-Obvious Notes for Next Session

1. **test_strong_penalties has a hardcoded API token** in `run/test_strong_penalties/config.yaml`. Do not commit or share this file. The token should be in `.env` only.

2. **Advantage2_system1.11 is Zephyr topology** (not Pegasus). Zephyr has different connectivity than the Pegasus used in all trial runs. Embedding behavior may differ slightly — first embedding may take longer on a new topology. The `timeout=600s` in the config handles this.

3. **The generalized_tuning job output dir** was named `results/generalized_tuning_20260225_134657/`. If the job completed successfully, this dir will exist on Sycamore. If it failed, it may not exist (the bug). Check with `ls -la ~/quantum_folding_2d/results/` after job ends.

4. **run/run_test_dwave metrics_summary.csv only has N=4,5 data** — N=6,7,8 data appears absent (may not have completed or wasn't in the CSV). This doesn't affect our config decisions since the key chain break insight came from N=4-5.

5. **Phase 1 of paper_exp4 is our-encoding-only by design**. Grid comparison is Phase 2 (gated). Don't add grid to the config until Phase 1 validates the hypothesis.

6. **ripple_carry_symbreak_qpu preset uses ripple_carry_symbreak as base** (with symbreak ConstraintSpec added). The untuned `ripple_carry_symbreak` is what `run_test_dwave` used for `rc_4`. Our QPU preset is a hardened version of the SA-tuned preset (`ripple_carry_symbreak_tuned`), not the untuned base.
