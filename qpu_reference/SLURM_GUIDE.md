# Claude Guide: slurm/ - HPC Cluster Scripts

## Purpose

SLURM batch scripts for running experiments on VCU's Sycamore HPC cluster. These scripts handle job submission, resource allocation, and environment setup.

**Related docs**: `../docs/SYCAMORE_HPC_GUIDE.md` (connection details, general cluster usage)

## When to Use

Work in slurm/ when:
- Submitting jobs to Sycamore cluster
- Debugging cluster execution issues
- Creating new batch scripts for experiments
- Modifying resource allocations

DO NOT use slurm/ for:
- **Local execution** → run scripts directly
- **Cluster connection** → see SYCAMORE_HPC_GUIDE.md
- **Experiment code** → use `../experiments/` or `../benchmarks/`

## Key Files

| File | Purpose | Typical Use |
|------|---------|-------------|
| **focal_tuning.sbatch** | Run focal_tuning.py on cluster | Parameter tuning |
| **generalized_tuning.sbatch** | Run generalized_tuning.py | Multi-scale tuning |
| **pipeline_job.sbatch** | Run benchmark pipeline | Systematic experiments |
| **.sbatch** | Various experiment scripts | Specific runs |

## Cluster: VCU Sycamore

**Connection**: `sycamore.cs.vcu.edu`
**Username**: VCU ID (e.g., `minkoffjg`)
**Project root**: `~/quantum_folding_2d/`

See `../docs/SYCAMORE_HPC_GUIDE.md` for SSH/SCP details.

## Typical Workflow

### 1. Upload Code

```bash
# From local machine
scp experiments/focal_tuning.py minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/experiments/
scp src/constraint_catalog.py minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/src/
scp slurm/focal_tuning.sbatch minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/slurm/
```

### 2. Fix Line Endings (if needed)

**Issue**: Windows CRLF line endings break bash scripts

**Solution** (on cluster):
```bash
sed -i 's/\r$//' slurm/focal_tuning.sbatch
```

**Why not dos2unix?**: Not installed on Sycamore

### 3. Submit Job

```bash
# SSH to cluster
ssh minkoffjg@sycamore.cs.vcu.edu
cd ~/quantum_folding_2d

# Submit
sbatch slurm/focal_tuning.sbatch
```

### 4. Monitor

```bash
# Check job status
squeue -u minkoffjg

# Watch (updates every 2s)
watch -n 2 squeue -u minkoffjg

# View output (while running)
tail -f slurm-<jobid>.out
```

### 5. Download Results

```bash
# From local machine
scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/results/focal_tuning_* ./results/
```

## SBATCH Script Structure

### Standard Template

```bash
#!/bin/bash
#SBATCH --job-name=my_experiment
#SBATCH --output=slurm-%j.out      # %j = job ID
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40         # Parallelism
#SBATCH --time=48:00:00            # Max runtime
#SBATCH --mem=64G                  # Memory

# Activate venv (CRITICAL!)
source ~/quantum_folding_2d/venv/bin/activate

# Run experiment
python experiments/my_script.py \
    --num-cpus 40 \
    --output results/my_output

# Deactivate
deactivate
```

### Resource Guidelines

| Experiment Type | CPUs | Memory | Time |
|-----------------|------|--------|------|
| Focal tuning (100 configs) | 40 | 32G | 24h |
| Generalized tuning (Phase 1) | 40 | 64G | 48h |
| Pipeline (small N) | 20 | 16G | 12h |
| Pipeline (N≥10) | 40 | 64G | 48h+ |

## Non-Obvious Quirks

### 1. SLURM Output Files Land in Project Root

**Expected**: `slurm/slurm-12345.out`
**Actual**: `~/quantum_folding_2d/slurm-12345.out`

**Cause**: `#SBATCH --output` path doesn't include `slurm/` subdirectory

**Fix**: Either live with it, or update scripts:
```bash
#SBATCH --output=slurm/slurm-%j.out
```

### 2. dos2unix Not Installed

**Problem**: Can't use `dos2unix` to fix Windows line endings

**Workaround**: `sed -i 's/\r$//' <file>`

**Prevention**: Configure git to auto-convert:
```bash
git config --global core.autocrlf input
```

### 3. Old src/generator.py Missing random_mj

**History**: Server code was stale, missing `random_mj()` method

**Symptom**: Job 14086 (first submission) failed with `AttributeError`

**Solution**: Always upload latest `src/generator.py` before submitting

**Checklist before cluster run**:
- [ ] Upload `src/generator.py`
- [ ] Upload `src/constraint_catalog.py` (if presets changed)
- [ ] Upload experiment script
- [ ] Upload .sbatch file
- [ ] Fix line endings (`sed -i 's/\r$//'`)

### 4. venv Activation is CRITICAL

**Must include**:
```bash
source ~/quantum_folding_2d/venv/bin/activate
```

**If missing**: Job uses system Python, missing dependencies → crash

### 5. UTF-8 Encoding Issues

**Problem**: Python scripts with emoji or special chars may crash with:
```
UnicodeEncodeError: 'ascii' codec can't encode character
```

**Fix**: Add to script header:
```python
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

Or ensure `update_timeline.py` etc. use `encoding="utf-8"` in `open()` calls.

## Checking Job Status

### View Queue

```bash
squeue -u minkoffjg
```

**Output**:
```
JOBID   USER     NAME           ST  TIME  NODES
123456  minkoffj focal_tuning   R   2:34  1
```

**ST codes**:
- `PD`: Pending (waiting for resources)
- `R`: Running
- `CG`: Completing
- `CD`: Completed

### View Job Details

```bash
scontrol show job 123456
```

### Cancel Job

```bash
scancel 123456
```

### View Output

```bash
# While running
tail -f ~/quantum_folding_2d/slurm-123456.out

# After completion
cat ~/quantum_folding_2d/slurm-123456.out
```

## Common Issues

### Issue: "Permission denied" on .sbatch file

**Cause**: File not executable (shouldn't matter, but sometimes does)

**Fix**: `chmod +x slurm/focal_tuning.sbatch`

### Issue: "command not found: python"

**Cause**: venv not activated

**Fix**: Add `source venv/bin/activate` to script

### Issue: Job stays PD (pending) forever

**Cause**: Insufficient resources available, or asking for too much

**Solution**: Check resource availability:
```bash
sinfo
```

Reduce CPU/mem request if needed.

### Issue: Script runs but produces no output

**Cause**: Error early in script (before main output), check stderr

**Fix**: `cat slurm-<jobid>.err`

## When to Update This Guide

Update this guide when you:
- Discover new cluster-specific issues
- Add new .sbatch templates
- Find workarounds for missing tools
- Change deployment procedures
- Encounter encoding or line-ending problems

Use `/savepoint` at session end to review changes.
