# Sycamore HPC Guide

This document covers connecting to VCU's Sycamore cluster and running SLURM jobs for the protein folding QUBO project.

## Connection Details

- **Hostname**: `sycamore.cs.vcu.edu`
- **Username**: Your VCU ID (e.g., `minkoffjg`)
- **Protocol**: SSH/SCP

## Uploading Files

### Single File Upload
```bash
scp path/to/local/file.py USERNAME@sycamore.cs.vcu.edu:~/quantum_folding_2d/path/to/destination/
```

**Examples:**
```bash
# Upload a Python script
scp experiments/focal_tuning.py minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/experiments/

# Upload a SLURM batch script
scp slurm/focal_tuning.sbatch minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/slurm/

# Upload updated source code
scp src/constraint_catalog.py minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/src/
```

### Upload Entire Repository
```bash
scp -r "C:\Users\jonah\Documents\Coding\quantum_folding_2d" minkoffjg@sycamore.cs.vcu.edu:~/
```

### Download Results
```bash
# Download a single file
scp minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/results/focal_tuning/summary.json ./results/

# Download entire results directory
scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/results/focal_tuning ./results/
```

## SSH Access

```bash
ssh minkoffjg@sycamore.cs.vcu.edu
cd ~/quantum_folding_2d
```

## Python Environment

The project uses a virtual environment on the server:

```bash
# Activate the venv (required before running Python scripts)
source venv/bin/activate

# Install new packages if needed
pip install tqdm  # example
```

**Important**: SLURM batch scripts must also activate the venv before running Python.

## SLURM Job Management

### Submitting Jobs
```bash
sbatch slurm/your_script.sbatch
```

### Check Job Status
```bash
# Show your queued/running jobs
squeue -u minkoffjg

# Watch job status (updates every 2 seconds)
watch -n 2 squeue -u minkoffjg

# Detailed job info
scontrol show job <JOB_ID>

# Job history for today
sacct -u minkoffjg --starttime=today --format=JobID,JobName,State,ExitCode,Reason
```

### Monitor Output
```bash
# Watch output in real-time
tail -f slurm/job_name_*.out

# Check recent output files
ls -lt slurm/ | head -10
```

### Cancel Jobs
```bash
scancel <JOB_ID>

# Cancel all your jobs
scancel -u minkoffjg
```

## Common Issues

### DOS Line Endings
Windows creates files with `\r\n` line endings, but SLURM requires Unix `\n` endings.

**Error:**
```
sbatch: error: Batch script contains DOS line breaks (\r\n)
```

**Fix (on server):**
```bash
dos2unix slurm/your_script.sbatch
# or
sed -i 's/\r$//' slurm/your_script.sbatch
```

### Job Fails Immediately (Exit Code 1)
Usually means a Python error. Debug by running manually:

```bash
source venv/bin/activate
python experiments/your_script.py --help
```

Common causes:
- Missing module (run `pip install <module>`)
- venv not activated in sbatch script
- Updated source files not uploaded to server

### Hostname Resolution Failed
Use the full hostname: `sycamore.cs.vcu.edu` (not just `sycamore` or `sycamore.hpc.vcu.edu`)

## SLURM Batch Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=slurm/my_job_%j.out
#SBATCH --error=slurm/my_job_%j.err
#SBATCH --time=7-00:00:00          # 7 days max
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@vcu.edu

cd ~/quantum_folding_2d
source venv/bin/activate           # IMPORTANT: activate venv!

python experiments/your_script.py --args here
```

## Typical Workflow

1. **Develop locally** - write/edit scripts on Windows
2. **Push changes** - `git push` from local
3. **SSH to server** - `ssh minkoffjg@sycamore.cs.vcu.edu`
4. **Pull on server** - `cd ~/quantum_folding_2d && git pull`
5. **Test manually first** - `source venv/bin/activate && python script.py --quick-test`
6. **Submit job** - `sbatch run/submit.sh <run_name>` (or run phases manually with nohup)
7. **Monitor progress** - `squeue -u minkoffjg` and `tail -f run/<run_name>/slurm_*.out`
8. **Download results** - `scp -r` analysis/solutions back to local machine

---

## D-Wave QPU Pipeline

### Prerequisites

1. **API token**: Set `DWAVE_API_TOKEN` in `~/quantum_folding_2d/.env` on the server:
   ```
   DWAVE_API_TOKEN=DEV-your-token-here
   ```
   Never commit the token. The pipeline reads it from the environment automatically.

2. **Check available solvers** (free, no QPU credits consumed):
   ```bash
   source venv/bin/activate
   dwave solvers --list
   ```
   Currently available: `Advantage_system4.1` (Pegasus) and `Advantage2_system1.11` (Zephyr).
   Use `Advantage2_system1.11` for paper experiments (newer topology, better performance).

3. **Install python-dotenv** if not present:
   ```bash
   pip install python-dotenv
   ```

### QPU Run Workflow (Step by Step)

```bash
ssh minkoffjg@sycamore.cs.vcu.edu
cd ~/quantum_folding_2d
git pull
source venv/bin/activate

# Run phases 1 & 2 (instance generation + QUBO stats — free, no QPU)
python -m benchmarks.pipeline.runner run/<run_name> --phases 1,2

# Verify instance count matches config (check for stale instance files from old runs)
ls run/<run_name>/instances/

# Run phase 3 (QPU solve — consumes Leap credits)
# Use nohup for long runs (disconnects safely)
nohup python -m benchmarks.pipeline.runner run/<run_name> --phases 3 \
    > run/<run_name>/phase3.log 2>&1 &
tail -f run/<run_name>/phase3.log

# Run phase 4 (analysis — free, local)
python -m benchmarks.pipeline.runner run/<run_name> --phases 4
```

### Pull Results Locally

```bash
# On local machine:
scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/run/<run_name>/solutions/ ./run/<run_name>/
scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/run/<run_name>/analysis/ ./run/<run_name>/
```

Then view in dashboard:
```bash
python -m streamlit run benchmarks/dashboard/app.py
```

### QPU Cost Tracking

Each solution JSON contains a `qpu_cost` block:
```json
"qpu_cost": {
  "qpu_access_time_us": 34530,
  "qpu_access_time_s": 0.034530,
  "leap_seconds_used": 0.034530,
  "num_reads": 100,
  "annealing_time_us": 20
}
```
`leap_seconds_used` is the D-Wave billing unit. Free tier = ~60 seconds/month.
At 20µs anneal + 100 reads: ~0.035s per task. At 1000 reads: ~0.35s per task.

### Common QPU Issues

**Stale instance files**: If phase 2 generates more instances than expected, check for leftover
JSON files from old runs in `run/<run_name>/instances/`. Delete extras and re-run phase 2.

**Stale solution files**: Same issue in `run/<run_name>/solutions/`. Delete any files not
matching the current config's formulations/instances before running phase 4.

**Token not found**: Verify `.env` exists with `DWAVE_API_TOKEN=...`. The pipeline uses
`python-dotenv` to load it — if dotenv isn't installed, token won't be read.

**Solver unavailable**: Check `dwave solvers --list` before running. `Advantage_system6.4` is
NOT available — use `Advantage2_system1.11` for paper experiments.

**High chain breaks (>20%)**: Increase `chain_strength` manually or reduce penalty weights.
Can also clear the embedding cache and retry (different embedding may be found).

**0% valid samples on QPU**: Expected for test runs with 100 reads and SA-tuned weights.
QPU needs either more reads (1000+) or QPU-specific weight retuning (bump lambdas 2-3x).

### Embedding Cache

Embeddings are cached in `run/<run_name>/embeddings/` (if configured). Re-using a cached
embedding skips the minorminer search and saves ~1-10 minutes per task. The cache is keyed
by `task_id` (instance + formulation + solver). Clear cache if QUBO structure changes.
