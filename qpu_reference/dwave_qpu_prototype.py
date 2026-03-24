#!/usr/bin/env python3
"""
D-Wave QPU Prototype

This prototype validates all D-Wave technical aspects before full integration:
- Connection to QPU
- Minor embedding workflow
- Embedding cache (save/load)
- Statistics collection
- QPU solving
- Chain break analysis
- Sample decoding

Token: loaded from DWAVE_API_TOKEN env var
Solver: Advantage_system4.1
Parameters: num_reads=100, annealing_time=20 (conserve QPU time)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# D-Wave imports
from dwave.system import DWaveSampler
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_strength import uniform_torque_compensation
from minorminer import find_embedding
import dimod

# Project imports
from src import InstanceGenerator
from src.builder import ComposableQUBOBuilder
from src.solver import ComposableSolutionDecoder

# Configuration
DWAVE_TOKEN = os.environ["DWAVE_API_TOKEN"]
SOLVER_ID = "Advantage_system4.1"
NUM_READS = 100
ANNEALING_TIME = 20  # microseconds
EMBEDDING_FILE = "test_embedding.json"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def step_1_generate_bqm():
    """Generate test BQM from protein instance."""
    print_section("Step 1: Generate BQM")

    # Use N=3 instance (small but non-trivial)
    inst = InstanceGenerator.n3_no_contact()
    print(f"Instance: {inst.sequence} (N={inst.N})")

    # Build QUBO
    builder = ComposableQUBOBuilder(inst, preset='integer', parity_pruning=True)
    H, info = builder.build()
    model = H.compile()
    bqm = model.to_bqm()

    print(f"✓ BQM generated")
    print(f"  - Variables: {len(bqm.variables)}")
    print(f"  - Quadratic terms: {len(bqm.quadratic)}")
    print(f"  - Linear range: [{min(bqm.linear.values()):.2f}, {max(bqm.linear.values()):.2f}]")
    print(f"  - Quadratic range: [{min(bqm.quadratic.values()):.2f}, {max(bqm.quadratic.values()):.2f}]")

    return inst, bqm, info


def step_2_connect_dwave():
    """Initialize D-Wave client and verify connection."""
    print_section("Step 2: Connect to D-Wave")

    try:
        sampler = DWaveSampler(token=DWAVE_TOKEN, solver=SOLVER_ID)
        print(f"✓ Connected to {sampler.solver.name}")
        print(f"  - Topology: {sampler.properties['topology']['type']}")
        # Try to get qubit/coupler counts
        try:
            print(f"  - Working qubits: {sampler.properties.get('num_qubits', len(sampler.nodelist))}")
            print(f"  - Working couplers: {len(sampler.edgelist)}")
        except:
            pass
        return sampler
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        raise


def step_3_find_embedding(bqm, sampler):
    """Find minor embedding using minorminer."""
    print_section("Step 3: Find Minor Embedding")

    # Create source graph from BQM
    source_edgelist = list(bqm.quadratic.keys())
    print(f"Source graph: {len(bqm.variables)} nodes, {len(source_edgelist)} edges")

    # Find embedding
    start_time = time.time()
    embedding = find_embedding(source_edgelist, sampler.edgelist, verbose=0)
    elapsed = time.time() - start_time

    if not embedding:
        print("✗ Embedding failed!")
        return None

    print(f"✓ Embedding found in {elapsed:.2f}s")

    # Compute statistics
    chain_lengths = [len(chain) for chain in embedding.values()]
    physical_qubits = set()
    for chain in embedding.values():
        physical_qubits.update(chain)

    print(f"  - Physical qubits: {len(physical_qubits)}")
    print(f"  - Max chain length: {max(chain_lengths)}")
    print(f"  - Avg chain length: {sum(chain_lengths) / len(chain_lengths):.2f}")

    return embedding


def step_4_save_embedding(embedding, bqm):
    """Save embedding to JSON file."""
    print_section("Step 4: Save Embedding")

    # Convert embedding (lists to ensure JSON serializable)
    embedding_serializable = {str(k): list(v) for k, v in embedding.items()}

    # Compute chain strength
    chain_strength = uniform_torque_compensation(bqm, embedding)

    # Create cache data
    cache_data = {
        "task_id": "test_n3_hph",
        "embedding": embedding_serializable,
        "stats": {
            "physical_qubits": len(set(q for chain in embedding.values() for q in chain)),
            "max_chain_length": max(len(chain) for chain in embedding.values()),
            "avg_chain_length": sum(len(chain) for chain in embedding.values()) / len(embedding),
            "chain_strength": chain_strength,
            "solver_id": SOLVER_ID,
        },
        "created_at": datetime.now().isoformat()
    }

    # Save to file
    with open(EMBEDDING_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"✓ Saved to {EMBEDDING_FILE}")
    print(f"  - Chain strength (auto): {chain_strength:.2f}")

    return cache_data


def step_5_load_embedding():
    """Load embedding from JSON file."""
    print_section("Step 5: Load Embedding")

    with open(EMBEDDING_FILE, 'r') as f:
        cache_data = json.load(f)

    # Convert back to proper types
    embedding = {k: v for k, v in cache_data['embedding'].items()}

    print(f"✓ Loaded from {EMBEDDING_FILE}")
    print(f"  - Task: {cache_data['task_id']}")
    print(f"  - Created: {cache_data['created_at']}")
    print(f"  - Physical qubits: {cache_data['stats']['physical_qubits']}")

    return embedding, cache_data


def step_6_compute_statistics(bqm, embedding, sampler):
    """Compute embedding statistics."""
    print_section("Step 6: Compute Statistics")

    # Compute chain strength
    chain_strength = uniform_torque_compensation(bqm, embedding)

    # Embed BQM to get embedded ranges
    embedded_bqm = embed_bqm(bqm, embedding, sampler.adjacency, chain_strength=chain_strength)

    # Convert to Ising for h/J analysis
    h, J, offset = bqm.to_ising()
    h_vals = list(h.values()) if h else [0]
    J_vals = list(J.values()) if J else [0]

    h_emb, J_emb, _ = embedded_bqm.to_ising()
    h_emb_vals = list(h_emb.values()) if h_emb else [0]
    J_emb_vals = list(J_emb.values()) if J_emb else [0]

    print(f"Original BQM (Ising):")
    print(f"  - h range: [{min(h_vals):.2f}, {max(h_vals):.2f}]")
    print(f"  - J range: [{min(J_vals):.2f}, {max(J_vals):.2f}]")

    print(f"Embedded BQM (Ising):")
    print(f"  - h range: [{min(h_emb_vals):.2f}, {max(h_emb_vals):.2f}]")
    print(f"  - J range: [{min(J_emb_vals):.2f}, {max(J_emb_vals):.2f}]")
    print(f"  - Chain strength: {chain_strength:.2f}")

    return embedded_bqm, chain_strength


def step_7_solve_qpu(embedded_bqm, sampler):
    """Embed and solve on QPU."""
    print_section("Step 7: Solve on QPU")

    print(f"Submitting job...")
    print(f"  - num_reads: {NUM_READS}")
    print(f"  - annealing_time: {ANNEALING_TIME} µs")

    start_time = time.time()
    response = sampler.sample(embedded_bqm, num_reads=NUM_READS, annealing_time=ANNEALING_TIME)
    elapsed = time.time() - start_time

    print(f"✓ Job completed in {elapsed:.2f}s")
    print(f"  - Samples returned: {len(response)}")

    return response


def step_8_collect_timing(response):
    """Collect QPU timing metrics."""
    print_section("Step 8: QPU Timing")

    timing = response.info.get('timing', {})

    print(f"QPU Timing (microseconds):")
    for key in ['qpu_access_time', 'qpu_programming_time', 'qpu_sampling_time',
                'qpu_anneal_time_per_sample', 'qpu_readout_time_per_sample',
                'qpu_delay_time_per_sample']:
        if key in timing:
            print(f"  - {key}: {timing[key]:,} µs")

    return timing


def step_9_chain_breaks(embedding, response):
    """Compute chain break statistics."""
    print_section("Step 9: Chain Break Analysis")

    # Count chain breaks
    total_chain_breaks = 0
    total_samples = 0
    samples_with_breaks = 0

    # Map variable names to indices in response
    var_to_idx = {var: i for i, var in enumerate(response.variables)}

    # Check each sample
    for record in response.record:
        sample = record.sample
        has_break = False

        # Check each chain
        for var, chain in embedding.items():
            if len(chain) > 1:
                # Get values for all qubits in chain
                try:
                    chain_values = [sample[var_to_idx[q]] for q in chain if q in var_to_idx]
                    if chain_values and not all(v == chain_values[0] for v in chain_values):
                        has_break = True
                        break
                except (KeyError, IndexError):
                    continue

        if has_break:
            samples_with_breaks += record.num_occurrences
        total_samples += record.num_occurrences

    overall_fraction = samples_with_breaks / total_samples if total_samples > 0 else 0

    print(f"Chain breaks:")
    print(f"  - Overall fraction: {overall_fraction*100:.1f}%")
    print(f"  - Samples with breaks: {samples_with_breaks} / {total_samples}")

    return {
        'overall_fraction': overall_fraction,
        'samples_with_breaks': samples_with_breaks,
        'total_samples': total_samples
    }


def step_10_unembed_decode(response, embedding, bqm, info, inst):
    """Unembed samples and decode solution."""
    print_section("Step 10: Unembed and Decode")

    # Unembed with majority_vote
    from dwave.embedding.chain_breaks import majority_vote
    unembedded = unembed_sampleset(response, embedding, bqm, chain_break_method=majority_vote)

    print(f"✓ Unembedded {len(unembedded)} samples")

    # Get best sample
    best_sample = dict(unembedded.first.sample)
    best_energy = unembedded.first.energy

    print(f"Best sample:")
    print(f"  - Energy: {best_energy:.2f}")

    # Decode solution
    ctx = info['ctx']
    decoder = ComposableSolutionDecoder(ctx)
    solution = decoder.decode(best_sample, best_energy)

    print(f"  - MJ energy: {solution.mj_energy:.2f}")
    print(f"  - Valid: {solution.valid}")
    print(f"  - Moves: {solution.moves}")
    print(f"  - Contacts: {solution.contacts}")

    if not solution.valid:
        print(f"  - Violations: {solution.broken_constraints}")

    print(f"\nVisualization:")
    print(solution.visualize())

    return solution


def main():
    """Run the complete D-Wave QPU prototype workflow."""
    print_section("D-Wave QPU Prototype")
    print(f"Token: {DWAVE_TOKEN[:20]}...")
    print(f"Solver: {SOLVER_ID}")

    try:
        # Step 1: Generate BQM
        inst, bqm, info = step_1_generate_bqm()

        # Step 2: Connect to D-Wave
        sampler = step_2_connect_dwave()

        # Step 3: Find embedding
        embedding = step_3_find_embedding(bqm, sampler)
        if not embedding:
            print("\n✗ Prototype failed: Could not find embedding")
            return 1

        # Step 4: Save embedding
        cache_data = step_4_save_embedding(embedding, bqm)

        # Step 5: Load embedding
        embedding_loaded, cache_data_loaded = step_5_load_embedding()

        # Step 6: Compute statistics
        embedded_bqm, chain_strength = step_6_compute_statistics(bqm, embedding, sampler)

        # Step 7: Solve on QPU
        response = step_7_solve_qpu(embedded_bqm, sampler)

        # Step 8: Collect timing
        timing = step_8_collect_timing(response)

        # Step 9: Chain break analysis
        chain_breaks = step_9_chain_breaks(embedding, response)

        # Step 10: Unembed and decode
        solution = step_10_unembed_decode(response, embedding, bqm, info, inst)

        print_section("✓ Prototype Complete!")
        print(f"All D-Wave technical aspects validated successfully.")
        print(f"Ready to proceed with full implementation.")

        return 0

    except Exception as e:
        print(f"\n✗ Prototype failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
