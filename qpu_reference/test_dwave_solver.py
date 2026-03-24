#!/usr/bin/env python3
"""
Test DWaveQPUSolver Class

This script tests the DWaveQPUSolver class by:
1. Generating a BQM
2. Solving with DWaveQPUSolver (embedding + solve + statistics)
3. Verifying all outputs
4. Testing cache reuse
"""

import os
from pathlib import Path

# Project imports
from src import InstanceGenerator, DWaveQPUSolver
from src.builder import ComposableQUBOBuilder
from src.solver import ComposableSolutionDecoder

# Configuration
DWAVE_TOKEN = os.environ["DWAVE_API_TOKEN"]
SOLVER_ID = "Advantage_system4.1"
NUM_READS = 100
ANNEALING_TIME = 20
CACHE_DIR = Path("test_solver_cache")
TASK_ID = "test_n3_solver"


def test_dwave_solver():
    """Test the complete DWaveQPUSolver workflow."""
    print("="*60)
    print("Testing DWaveQPUSolver Class")
    print("="*60)

    # Step 1: Generate BQM
    print("\n1. Generating BQM...")
    inst = InstanceGenerator.n3_no_contact()
    builder = ComposableQUBOBuilder(inst, preset='integer', parity_pruning=True)
    H, info = builder.build()
    model = H.compile()
    bqm = model.to_bqm()
    print(f"✓ BQM: {len(bqm.variables)} variables, {len(bqm.quadratic)} quadratic terms")

    # Step 2: Create solver
    print("\n2. Creating DWaveQPUSolver...")
    solver = DWaveQPUSolver(
        token=DWAVE_TOKEN,
        solver_id=SOLVER_ID,
        embedding_cache_dir=CACHE_DIR
    )
    print(f"✓ Solver created with cache dir: {CACHE_DIR}")

    # Step 3: Solve (first time - no cache)
    print("\n3. Solving (first time - will find embedding)...")
    params = {
        'num_reads': NUM_READS,
        'annealing_time': ANNEALING_TIME,
        'chain_strength': None,  # Auto
        'find_embedding_params': {
            'timeout': 60,
            'tries': 10,
        }
    }
    result = solver.solve(bqm, TASK_ID, params)
    print(f"✓ Solved successfully")

    # Step 4: Verify result structure
    print("\n4. Verifying result structure...")
    required_keys = ['sampleset', 'timing', 'embedding_info', 'chain_breaks']
    for key in required_keys:
        if key not in result:
            print(f"✗ Missing key: {key}")
            return False
        print(f"✓ Has key: {key}")

    # Step 5: Verify sampleset
    print("\n5. Verifying sampleset...")
    sampleset = result['sampleset']
    print(f"✓ Samples: {len(sampleset)}")
    print(f"  - Best energy: {sampleset.first.energy:.2f}")
    print(f"  - Variables: {len(sampleset.variables)}")

    # Step 6: Verify timing
    print("\n6. Verifying QPU timing...")
    timing = result['timing']
    timing_keys = ['qpu_access_time', 'qpu_programming_time', 'qpu_sampling_time']
    for key in timing_keys:
        if key in timing:
            print(f"✓ {key}: {timing[key]:,} µs")

    # Step 7: Verify embedding info
    print("\n7. Verifying embedding info...")
    emb_info = result['embedding_info']
    print(f"✓ Physical qubits: {emb_info['physical_qubits']}")
    print(f"✓ Max chain length: {emb_info['max_chain_length']}")
    print(f"✓ Avg chain length: {emb_info['avg_chain_length']:.2f}")
    print(f"✓ Chain strength: {emb_info['chain_strength']:.2f}")
    print(f"✓ Cached: {emb_info['cached']}")

    if emb_info['cached']:
        print("✗ Expected cached=False for first solve")
        return False

    # Step 8: Verify chain breaks
    print("\n8. Verifying chain breaks...")
    chain_breaks = result['chain_breaks']
    print(f"✓ Overall fraction: {chain_breaks['overall_fraction']*100:.1f}%")
    print(f"✓ Samples with breaks: {chain_breaks['samples_with_breaks']} / {chain_breaks['total_samples']}")

    # Step 9: Decode solution
    print("\n9. Decoding solution...")
    ctx = info['ctx']
    decoder = ComposableSolutionDecoder(ctx)
    solution = decoder.decode(dict(sampleset.first.sample), sampleset.first.energy)
    print(f"✓ Solution decoded:")
    print(f"  - MJ energy: {solution.mj_energy:.2f}")
    print(f"  - Valid: {solution.valid}")
    print(f"  - Moves: {solution.moves}")

    # Step 10: Solve again (should use cache)
    print("\n10. Solving again (should use cache)...")
    result2 = solver.solve(bqm, TASK_ID, params)
    emb_info2 = result2['embedding_info']
    print(f"✓ Cached: {emb_info2['cached']}")

    if not emb_info2['cached']:
        print("✗ Expected cached=True for second solve")
        return False

    # Verify embedding stats match
    if emb_info['physical_qubits'] != emb_info2['physical_qubits']:
        print(f"✗ Physical qubits mismatch")
        return False
    print(f"✓ Embedding stats match (physical qubits: {emb_info2['physical_qubits']})")

    # Cleanup
    print("\n11. Cleanup...")
    cache_file = CACHE_DIR / f"{TASK_ID}.json"
    if cache_file.exists():
        cache_file.unlink()
    if CACHE_DIR.exists() and not list(CACHE_DIR.iterdir()):
        CACHE_DIR.rmdir()
    print("✓ Test files cleaned up")

    print("\n" + "="*60)
    print("✓ All DWaveQPUSolver Tests Passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_dwave_solver()
    exit(0 if success else 1)
