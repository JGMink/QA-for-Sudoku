# QPU Reference: Sycamore + D-Wave

Reference files from the `quantum_folding_2d` project. Covers three topics.

---

## 1. Connect to Sycamore, run stuff, grab results

| File | What it covers |
|------|---------------|
| `SYCAMORE_HPC_GUIDE.md` | **Main guide**: SSH/SCP commands, SLURM job management, QPU pipeline workflow step-by-step, cost tracking, common errors |
| `SLURM_GUIDE.md` | Detailed SLURM guide: sbatch template, resource guidelines, non-obvious quirks (DOS line endings, venv activation, UTF-8 issues) |
| `tuned_benchmark.sbatch` | Real sbatch script example: shows partition, cpus, mem, time, venv activation, pipeline runner invocation |

**Key facts:**
- Hostname: `sycamore.cs.vcu.edu`, user: VCU ID
- Python env lives at `~/project/venv/` — must `source venv/bin/activate` in every sbatch
- Use `nohup python ... &` for long QPU runs (safe against SSH disconnect)
- Pull results with `scp -r user@sycamore.cs.vcu.edu:~/project/run/name/solutions/ ./`

---

## 2. Make and run QPU code with separate env

| File | What it covers |
|------|---------------|
| `dwave_qpu_prototype.py` | **Complete 10-step QPU workflow**: BQM generation → D-Wave connection → minor embedding → embedding cache save/load → statistics → QPU solve → timing collection → chain break analysis → unembed + decode |
| `test_dwave_solver.py` | Tests `DWaveQPUSolver` class: first solve (no cache), second solve (cache hit), verifies timing/embedding/chain-break output structure |
| `config_qpu.yaml` | Example YAML config for a QPU benchmark run: token from env var, solver ID, annealing_time, num_reads, chain_strength, embedding params |
| `requirements_dwave.txt` | Python dependencies: `dwave-ocean-sdk`, `pyqubo`, `dimod`, `dwave-samplers`, `python-dotenv` |

**Key facts:**
- Token goes in `.env` as `DWAVE_API_TOKEN=...`, loaded via `python-dotenv`
- Current working solver: `Advantage2_system1.11` (Zephyr, ~7000 qubits) — NOT `Advantage_system6.4` (unavailable)
- `chain_strength=null` → auto-computed via `uniform_torque_compensation` (recommended)
- Embeddings are cached to JSON; re-using saves 1–10 min per task
- Use `nohup` on Sycamore for phase 3 (QPU solve); phases 1, 2, 4 are free (no credits)

---

## 3. Estimate QPU time and cost

| File | What it covers |
|------|---------------|
| `QPU_EXPERIMENTS_SUMMARY.md` | Full report on 4 rounds of real QPU experiments: exact credit usage, timing data, chain break rates, what failed and why |
| `SESSION_20260226_qpu_prep.md` | Practical analysis: chain break thresholds, safe vs dangerous weight ranges, QPU weight strategy (1.33× bump, λ_adj reduction) |
| `qpu_statistics.py` | Code to extract QPU timing from solution JSONs: `qpu_access_time`, `qpu_programming_time`, `qpu_sampling_time`, chain break %, embedding stats |

**Key cost facts (from real experiments):**
- D-Wave billing unit: `leap_seconds_used` = `qpu_access_time_us / 1e6`
- Free Leap tier ≈ 60 seconds/month
- At 20 µs anneal + 100 reads: ~0.035 s/task (cheap)
- At 20 µs anneal + 1000 reads: ~0.35 s/task
- 54 tasks × 1000 reads ≈ 0.18 s total QPU time (well within free tier)
- **Chain breaks > 20% = danger zone** — means chain_strength is too low relative to QUBO J-range
- Programming time (~3–7 ms) is per-task overhead regardless of reads; batching reads is efficient

**Timing breakdown per QPU call (from `response.info['timing']`):**
- `qpu_access_time`: total time the QPU was reserved (billing unit)
- `qpu_programming_time`: time to load the problem onto the chip
- `qpu_sampling_time`: actual annealing time across all reads
- `qpu_anneal_time_per_sample`: annealing_time parameter (e.g., 20 µs)
- `qpu_readout_time_per_sample`: readout overhead per sample
