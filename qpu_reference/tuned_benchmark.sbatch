#!/bin/bash
#SBATCH --job-name=tuned_benchmark
#SBATCH --output=tuned_benchmark_%j.out
#SBATCH --error=tuned_benchmark_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

# Tuned Penalty Weight Benchmark
# Tests optimal weights from focal tuning experiment on N=4-15 instances
#
# Workload estimate:
#   18 instances × 3 formulations × 2 solvers = 108 jobs
#   @ ~30s each = ~54 min serial, ~5-10 min with 16 CPUs

echo "========================================"
echo "TUNED PENALTY WEIGHT BENCHMARK"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Start time: $(date)"
echo "========================================"

# Change to project directory
cd ~/quantum_folding_2d

# Activate virtual environment
source venv/bin/activate

# Print Python info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "========================================"

# Run all 4 phases of the benchmark pipeline
python -m benchmarks.pipeline.runner run/tuned_benchmark --phases 1,2,3,4

EXIT_CODE=$?

echo "========================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

# Print summary if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Benchmark completed"
    echo ""
    echo "Results location: run/tuned_benchmark/analysis/"
    echo "  - metrics_long.csv: Detailed per-run metrics"
    echo "  - metrics_summary.csv: Aggregated statistics"
    echo ""
    echo "To download results:"
    echo "  scp -r minkoffjg@sycamore.cs.vcu.edu:~/quantum_folding_2d/run/tuned_benchmark/analysis ./run/tuned_benchmark/"
else
    echo "FAILED with exit code $EXIT_CODE"
    echo "Check error log: tuned_benchmark_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
