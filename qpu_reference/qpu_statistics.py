"""QPU statistics extraction and aggregation.

Functions to extract D-Wave QPU-specific statistics from solution JSONs.
"""

from typing import List
import pandas as pd


def extract_qpu_stats(solutions: List[dict]) -> pd.DataFrame:
    """Extract QPU statistics from solution JSONs.

    Args:
        solutions: List of solution dictionaries with QPU data

    Returns:
        DataFrame with columns:
            - task_id
            - instance_name
            - N (protein length)
            - formulation
            - physical_qubits
            - max_chain_length
            - avg_chain_length
            - chain_strength
            - chain_break_pct
            - samples_with_breaks
            - qpu_access_time_ms
            - qpu_programming_time_ms
            - qpu_sampling_time_ms
            - h_range (original)
            - J_range (original)
            - h_range_embedded
            - J_range_embedded
            - embedding_cached
    """
    rows = []

    for sol in solutions:
        # Skip if not a QPU solution
        if sol.get('solver', {}).get('solver_type') != 'dwave_qpu':
            continue

        # Extract N from instance name (e.g., "random_N5_s2" -> 5)
        inst_name = sol.get('instance_name', '')
        n = None
        if '_N' in inst_name:
            parts = inst_name.split('_')
            for part in parts:
                if part.startswith('N') and part[1:].isdigit():
                    n = int(part[1:])
                    break

        # Extract QPU timing (convert microseconds to milliseconds)
        qpu_timing = sol.get('qpu_timing', {})
        qpu_access_time_ms = qpu_timing.get('qpu_access_time', 0) / 1000.0
        qpu_programming_time_ms = qpu_timing.get('qpu_programming_time', 0) / 1000.0
        qpu_sampling_time_ms = qpu_timing.get('qpu_sampling_time', 0) / 1000.0

        # Extract embedding info
        embedding = sol.get('embedding', {})
        physical_qubits = embedding.get('physical_qubits', 0)
        max_chain_length = embedding.get('max_chain_length', 0)
        avg_chain_length = embedding.get('avg_chain_length', 0.0)
        chain_strength = embedding.get('chain_strength', 0.0)
        h_range = embedding.get('h_range', 0.0)
        J_range = embedding.get('J_range', 0.0)
        h_range_embedded = embedding.get('h_range_embedded', 0.0)
        J_range_embedded = embedding.get('J_range_embedded', 0.0)
        embedding_cached = embedding.get('cached', False)

        # Extract chain breaks
        chain_breaks = sol.get('chain_breaks', {})
        chain_break_pct = chain_breaks.get('overall_fraction', 0.0) * 100.0
        samples_with_breaks = chain_breaks.get('samples_with_breaks', 0)

        rows.append({
            'task_id': sol.get('task_id', ''),
            'instance_name': inst_name,
            'N': n,
            'formulation': sol.get('formulation', ''),
            'physical_qubits': physical_qubits,
            'max_chain_length': max_chain_length,
            'avg_chain_length': avg_chain_length,
            'chain_strength': chain_strength,
            'chain_break_pct': chain_break_pct,
            'samples_with_breaks': samples_with_breaks,
            'qpu_access_time_ms': qpu_access_time_ms,
            'qpu_programming_time_ms': qpu_programming_time_ms,
            'qpu_sampling_time_ms': qpu_sampling_time_ms,
            'h_range': h_range,
            'J_range': J_range,
            'h_range_embedded': h_range_embedded,
            'J_range_embedded': J_range_embedded,
            'embedding_cached': embedding_cached,
        })

    return pd.DataFrame(rows)


def compute_embedding_quality_score(stats: pd.DataFrame) -> pd.DataFrame:
    """Compute embedding quality metrics.

    Quality factors:
        - Chain break rate (lower is better, weight: 50%)
        - Max chain length (lower is better, weight: 25%)
        - Physical qubit efficiency (logical/physical ratio, higher is better, weight: 25%)

    Args:
        stats: DataFrame from extract_qpu_stats()

    Returns:
        DataFrame with original stats plus 'score' column (0-100)
    """
    result = stats.copy()

    # Normalize chain break rate (0-100% -> 100-0 score)
    chain_break_score = 100 - result['chain_break_pct']
    chain_break_score = chain_break_score.clip(lower=0)

    # Normalize max chain length (1-20 range, lower is better)
    # Score = 100 when max_chain=1, score = 0 when max_chain >= 20
    max_chain_score = 100 * (1 - (result['max_chain_length'] - 1) / 19.0)
    max_chain_score = max_chain_score.clip(lower=0, upper=100)

    # Physical qubit efficiency
    # Estimate logical qubits from QUBO variables (assume ~300 for N=5)
    # For simplicity, use a rough estimate: N^2 * 10 variables
    estimated_logical = result['N'] ** 2 * 10
    efficiency = (estimated_logical / result['physical_qubits']) * 100
    efficiency_score = efficiency.clip(upper=100)

    # Weighted combination
    result['score'] = (
        0.50 * chain_break_score +
        0.25 * max_chain_score +
        0.25 * efficiency_score
    )

    return result


def aggregate_by_solver(stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate QPU stats by solver type.

    Args:
        stats: DataFrame from extract_qpu_stats()

    Returns:
        DataFrame grouped by formulation with aggregate statistics
    """
    if stats.empty:
        return pd.DataFrame()

    agg_dict = {
        'task_id': 'count',
        'physical_qubits': ['mean', 'median', 'std'],
        'max_chain_length': ['mean', 'median', 'std', 'max'],
        'avg_chain_length': ['mean', 'median', 'std'],
        'chain_strength': ['mean', 'median', 'std'],
        'chain_break_pct': ['mean', 'median', 'std'],
        'qpu_access_time_ms': ['mean', 'median', 'std'],
        'qpu_programming_time_ms': ['mean', 'median', 'std'],
        'qpu_sampling_time_ms': ['mean', 'median', 'std'],
        'embedding_cached': 'mean',  # Cache hit rate
    }

    grouped = stats.groupby('formulation').agg(agg_dict)
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    grouped = grouped.rename(columns={'task_id_count': 'n_tasks'})
    grouped = grouped.rename(columns={'embedding_cached_mean': 'cache_hit_rate'})

    return grouped.reset_index()
