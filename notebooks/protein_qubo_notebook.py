"""
Lattice Protein Folding with Quantum Annealing
================================================
This notebook demonstrates how to formulate the lattice protein folding problem
as a QUBO (Quadratic Unconstrained Binary Optimization) problem suitable for
D-Wave quantum annealers.

Based on: Irbäck et al., "Folding lattice proteins confined on minimal grids 
using a quantum-inspired encoding" (2025)
"""

# %% [markdown]
# # 1. Introduction
# 
# ## The Problem
# We want to find the minimum energy structure of a protein on a lattice grid.
# The protein is represented as a chain of amino acids, where:
# - Each amino acid must occupy exactly one lattice site
# - No two amino acids can occupy the same site (self-avoidance)
# - Consecutive amino acids must be adjacent on the lattice (connectivity)
# - Non-bonded amino acids that are adjacent on the lattice contribute to the energy
#
# ## The QUBO Approach
# We use a "fieldlike" binary encoding where we have a binary variable b_{i,n}
# for each amino acid i and lattice site n:
# - b_{i,n} = 1 if amino acid i is at site n
# - b_{i,n} = 0 otherwise
#
# The total energy is:
# E = E_MJ + λ₁·E₁ + λ₂·E₂ + λ₃·E₃
#
# where:
# - E_MJ: Miyazawa-Jernigan interaction energy (what we want to minimize)
# - E₁: Penalty ensuring each amino acid occupies exactly one site
# - E₂: Penalty ensuring each site has at most one amino acid
# - E₃: Penalty ensuring chain connectivity

# %%
import numpy as np
from pyqubo import Array, Constraint, Placeholder
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx

# %% [markdown]
# # 2. Problem Setup

# %%
class LatticeProteinQUBO:
    """
    Class to formulate lattice protein folding as a QUBO problem
    """
    
    def __init__(self, chain, adjacency_matrix, interaction_matrix, 
                 lambda1=1.0, lambda2=1.0, lambda3=1.0):
        """
        Parameters:
        -----------
        chain : list
            List of amino acid types (e.g., ['H', 'P', 'C', 'H'])
        adjacency_matrix : np.array
            Adjacency matrix of the lattice (1 if sites are neighbors, 0 otherwise)
        interaction_matrix : dict
            Dictionary of interaction energies C(a_i, a_j)
        lambda1, lambda2, lambda3 : float
            Lagrange penalty parameters
        """
        self.chain = chain
        self.N = len(chain)
        self.adj = adjacency_matrix
        self.C = interaction_matrix
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        # Validate inputs
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1] == self.N, \
            "Adjacency matrix must be N×N where N is chain length"
    
    def build_qubo(self):
        """
        Build the QUBO formulation using PyQUBO
        
        Returns:
        --------
        model : pyqubo.Model
            The compiled QUBO model
        """
        N = self.N
        
        # Create binary variables b[i,n] for amino acid i at site n
        b = Array.create('b', shape=(N, N), vartype='BINARY')
        
        # === E_MJ: Interaction Energy ===
        # E_MJ = -sum_{|i-j|>1} C(a_i, a_j) * sum_{<n,m>} b_{i,n} * b_{j,m} * adj[n,m]
        E_MJ = 0
        for i in range(N):
            for j in range(i+2, N):  # |i-j| > 1
                a_i, a_j = self.chain[i], self.chain[j]
                c_ij = self.C.get((a_i, a_j), 0.0)
                if c_ij != 0:
                    for n in range(N):
                        for m in range(N):
                            if self.adj[n, m] == 1:
                                E_MJ += -c_ij * b[i, n] * b[j, m]
        
        # === E1: Each amino acid at exactly one site ===
        # E_1 = sum_i (sum_n b_{i,n} - 1)^2
        E1 = 0
        for i in range(N):
            constraint_sum = sum(b[i, n] for n in range(N)) - 1
            E1 += constraint_sum ** 2
        
        # === E2: Each site has at most one amino acid ===
        # E_2 = 0.5 * sum_n sum_{i!=j} b_{i,n} * b_{j,n}
        E2 = 0
        for n in range(N):
            for i in range(N):
                for j in range(i+1, N):  # Only count pairs once
                    E2 += b[i, n] * b[j, n]
        
        # === E3: Chain connectivity ===
        # E_3 = sum_{i=1}^{N-1} sum_n b_{i,n} * sum_{m: not adjacent to n} b_{i+1,m}
        E3 = 0
        for i in range(N - 1):
            for n in range(N):
                for m in range(N):
                    # Penalize if consecutive amino acids are not adjacent
                    if self.adj[n, m] == 0 and n != m:
                        E3 += b[i, n] * b[i+1, m]
        
        # === Total Energy ===
        # Using Placeholder for lambdas allows us to change them without recompiling
        H = E_MJ + Placeholder('lambda1') * E1 + \
            Placeholder('lambda2') * E2 + Placeholder('lambda3') * E3
        
        # Compile the model
        model = H.compile()
        
        return model
    
    def solve_qubo(self, model, feed_dict=None):
        """
        Get the QUBO matrix from the model
        
        Parameters:
        -----------
        model : pyqubo.Model
            Compiled QUBO model
        feed_dict : dict, optional
            Dictionary to feed placeholder values
        
        Returns:
        --------
        qubo : dict
            QUBO dictionary {(i,j): value}
        offset : float
            Constant offset
        """
        if feed_dict is None:
            feed_dict = {
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'lambda3': self.lambda3
            }
        
        bqm = model.to_bqm(feed_dict=feed_dict)
        qubo, offset = bqm.to_qubo()
        
        return qubo, offset
    
    def decode_solution(self, sample):
        """
        Decode a binary solution to amino acid positions
        
        Parameters:
        -----------
        sample : dict
            Dictionary of binary variable assignments
        
        Returns:
        --------
        positions : list
            List of site indices for each amino acid
        b_matrix : np.array
            The N×N binary matrix
        """
        N = self.N
        b_matrix = np.zeros((N, N), dtype=int)
        positions = []
        
        for i in range(N):
            for n in range(N):
                var_name = f'b[{i}][{n}]'
                if var_name in sample:
                    b_matrix[i, n] = sample[var_name]
                    if sample[var_name] == 1:
                        positions.append(n)
        
        return positions, b_matrix

# %% [markdown]
# # 3. Energy Evaluation Module
# 
# This module evaluates the energy of any configuration (from the quantum annealer
# or other optimization methods).

# %%
class EnergyEvaluator:
    """
    Evaluate the energy of a protein configuration
    """
    
    def __init__(self, chain, adjacency_matrix, interaction_matrix,
                 lambda1=1.0, lambda2=1.0, lambda3=1.0):
        self.chain = chain
        self.N = len(chain)
        self.adj = adjacency_matrix
        self.C = interaction_matrix
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
    
    def compute_E_MJ(self, b, verbose=False):
        """Compute interaction energy"""
        N = self.N
        E = 0.0
        contacts = []
        
        for i in range(N):
            for j in range(i+2, N):
                a_i, a_j = self.chain[i], self.chain[j]
                c_ij = self.C.get((a_i, a_j), 0.0)
                if c_ij != 0:
                    for n in range(N):
                        for m in range(N):
                            if b[i, n] == 1 and b[j, m] == 1 and self.adj[n, m] == 1:
                                E += c_ij
                                contacts.append((i, j, a_i, a_j, n, m, c_ij))
                                if verbose:
                                    print(f"  Contact: amino acids {i}({a_i}) and {j}({a_j}) "
                                          f"at sites {n} and {m}, C={c_ij}")
        
        if verbose:
            print(f"  Total interaction sum: {E}")
            print(f"  E_MJ = -{E} = {-E}")
        
        return -E, contacts
    
    def compute_E1(self, b):
        """Ensure each amino acid at exactly one site"""
        return int(np.sum((np.sum(b, axis=1) - 1) ** 2))
    
    def compute_E2(self, b):
        """Ensure each site has at most one amino acid"""
        N = self.N
        E2 = 0.0
        for n in range(N):
            site_occupancy = np.sum(b[:, n])
            E2 += site_occupancy * (site_occupancy - 1)
        return int(0.5 * E2)
    
    def compute_E3(self, b):
        """Ensure chain connectivity"""
        N = self.N
        non_adj = 1 - self.adj - np.eye(N)
        E3 = 0.0
        for i in range(N - 1):
            E3 += b[i, :] @ non_adj @ b[i+1, :]
        return int(E3)
    
    def evaluate(self, b_matrix, verbose=False):
        """
        Evaluate total energy and all components
        
        Parameters:
        -----------
        b_matrix : np.array
            N×N binary matrix of amino acid positions
        verbose : bool
            If True, print detailed breakdown
        
        Returns:
        --------
        total_energy : float
            Total QUBO energy
        breakdown : tuple
            (E_MJ, E1, E2, E3)
        """
        if verbose:
            print("\nBit matrix b:")
            print(b_matrix)
            print("\nPositions:")
            for i in range(self.N):
                pos = np.where(b_matrix[i, :] == 1)[0]
                aa = self.chain[i]
                if len(pos) > 0:
                    print(f"  Amino acid {i} ({aa}): site {pos[0]}")
                else:
                    print(f"  Amino acid {i} ({aa}): NO SITE ASSIGNED")
            print("\nComputing E_MJ:")
        
        E_MJ, contacts = self.compute_E_MJ(b_matrix, verbose)
        E1 = self.compute_E1(b_matrix)
        E2 = self.compute_E2(b_matrix)
        E3 = self.compute_E3(b_matrix)
        
        total = E_MJ + self.lambda1 * E1 + self.lambda2 * E2 + self.lambda3 * E3
        
        if verbose:
            print(f"\nEnergy components:")
            print(f"  E_MJ = {E_MJ}")
            print(f"  E_1  = {E1}")
            print(f"  E_2  = {E2}")
            print(f"  E_3  = {E3}")
            print(f"\nTotal energy calculation:")
            print(f"  E = {E_MJ} + {self.lambda1}×{E1} + {self.lambda2}×{E2} + {self.lambda3}×{E3}")
            print(f"  E = {E_MJ} + {self.lambda1*E1} + {self.lambda2*E2} + {self.lambda3*E3}")
            print(f"  E = {total}")
            print(f"\nValid conformation: {self.is_valid((E_MJ, E1, E2, E3))}")
        
        return total, (E_MJ, E1, E2, E3), contacts
    
    @staticmethod
    def is_valid(breakdown):
        """Check if configuration satisfies all constraints"""
        _, E1, E2, E3 = breakdown
        return E1 == 0 and E2 == 0 and E3 == 0

# %% [markdown]
# # 4. Visualization Tools

# %%
def visualize_lattice(chain, positions, adjacency_matrix, title="Lattice Configuration"):
    """
    Visualize the protein on the lattice
    
    Parameters:
    -----------
    chain : list
        Amino acid sequence
    positions : list
        Site index for each amino acid
    adjacency_matrix : np.array
        Lattice adjacency matrix
    title : str
        Plot title
    """
    N = len(chain)
    
    # Create a graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, seed=42)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw lattice edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray', ax=ax)
    
    # Draw all lattice sites
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', 
                           node_size=800, alpha=0.5, ax=ax)
    
    # Highlight occupied sites with amino acids
    colors = {'H': 'red', 'C': 'blue', 'P': 'green'}
    
    if len(positions) == N:
        for i, site in enumerate(positions):
            aa = chain[i]
            color = colors.get(aa, 'yellow')
            nx.draw_networkx_nodes(G, pos, nodelist=[site], 
                                  node_color=color, node_size=1000, 
                                  alpha=0.8, ax=ax)
            # Add amino acid label
            x, y = pos[site]
            ax.text(x, y, f"{aa}{i}", fontsize=10, ha='center', 
                   va='center', fontweight='bold', color='white')
        
        # Draw chain connections
        chain_edges = [(positions[i], positions[i+1]) for i in range(N-1)]
        nx.draw_networkx_edges(G, pos, edgelist=chain_edges, 
                              width=3, edge_color='black', ax=ax)
    
    # Draw site labels
    labels = {i: str(i) for i in range(N)}
    label_pos = {k: (v[0], v[1]-0.15) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels, font_size=8, 
                           font_color='black', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors.get(aa, 'yellow'), 
                                 markersize=10, label=aa)
                      for aa in set(chain)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax

# %% [markdown]
# # 5. Example: Simple 4-Amino Acid Chain

# %%
# Define the problem
chain = ['H', 'P', 'C', 'H']
N = len(chain)

# Define lattice adjacency (square lattice: 0-1-2-3 in a line, with 0-2 also adjacent)
adjacency_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])

# Interaction matrix (HH favorable, CC unfavorable)
interaction_matrix = {
    ('H', 'H'): 1, ('C', 'C'): -1,
    ('H', 'C'): 0, ('C', 'H'): 0,
    ('H', 'P'): 0, ('P', 'H'): 0,
    ('C', 'P'): 0, ('P', 'C'): 0,
    ('P', 'P'): 0
}

# Lagrange parameters
lambda1, lambda2, lambda3 = 1.0, 1.0, 1.0

print("="*60)
print("LATTICE PROTEIN FOLDING PROBLEM")
print("="*60)
print(f"Chain: {chain}")
print(f"Chain length: {N}")
print(f"Lagrange parameters: λ₁={lambda1}, λ₂={lambda2}, λ₃={lambda3}")
print(f"\nInteraction matrix:")
for key, value in interaction_matrix.items():
    if value != 0:
        print(f"  C{key} = {value}")

# %% [markdown]
# # 6. Build QUBO Model

# %%
print("\n" + "="*60)
print("BUILDING QUBO MODEL")
print("="*60)

# Create QUBO formulation
protein_qubo = LatticeProteinQUBO(
    chain=chain,
    adjacency_matrix=adjacency_matrix,
    interaction_matrix=interaction_matrix,
    lambda1=lambda1,
    lambda2=lambda2,
    lambda3=lambda3
)

# Build the model
model = protein_qubo.build_qubo()
print("✓ Model compiled successfully")

# Get QUBO matrix
qubo, offset = protein_qubo.solve_qubo(model)
print(f"✓ QUBO matrix generated")
print(f"  Number of QUBO terms: {len(qubo)}")
print(f"  Constant offset: {offset}")

# Show a few QUBO terms as examples
print(f"\nSample QUBO terms:")
for i, (key, value) in enumerate(list(qubo.items())[:5]):
    print(f"  {key}: {value:.2f}")

# %% [markdown]
# # 7. Example Solution Evaluation

# %%
print("\n" + "="*60)
print("EXAMPLE: VALID CONFIGURATION")
print("="*60)

# Example valid configuration: chain follows path 1->0->2->3
bitstring = '0010100001000001'
b_matrix = np.array([list(map(int, bitstring[i*N:(i+1)*N])) for i in range(N)])

# Create evaluator
evaluator = EnergyEvaluator(
    chain=chain,
    adjacency_matrix=adjacency_matrix,
    interaction_matrix=interaction_matrix,
    lambda1=lambda1,
    lambda2=lambda2,
    lambda3=lambda3
)

# Evaluate energy
total_E, breakdown, contacts = evaluator.evaluate(b_matrix, verbose=True)

# Decode positions for visualization
positions = [np.where(b_matrix[i, :] == 1)[0][0] for i in range(N)]
print(f"\nChain path: {' -> '.join(map(str, positions))}")

# Visualize
fig, ax = visualize_lattice(chain, positions, adjacency_matrix, 
                            title=f"Valid Configuration (E={total_E})")
plt.show()

# %% [markdown]
# # 8. Example: Invalid Configuration

# %%
print("\n" + "="*60)
print("EXAMPLE: INVALID CONFIGURATION (Multiple Violations)")
print("="*60)

# Invalid configuration with multiple violations
bitstring_invalid = '1100010101001010'
b_matrix_invalid = np.array([list(map(int, bitstring_invalid[i*N:(i+1)*N])) 
                             for i in range(N)])

# Evaluate energy
total_E_invalid, breakdown_invalid, contacts_invalid = evaluator.evaluate(
    b_matrix_invalid, verbose=True)

# %% [markdown]
# # 9. Export QUBO for D-Wave

# %%
print("\n" + "="*60)
print("EXPORTING FOR D-WAVE")
print("="*60)

# The QUBO dictionary can be directly used with D-Wave's samplers
print("QUBO format ready for D-Wave Ocean SDK:")
print("""
from dwave.system import DWaveSampler, EmbeddingComposite

# Create sampler
sampler = EmbeddingComposite(DWaveSampler())

# Submit to D-Wave
sampleset = sampler.sample_qubo(qubo, num_reads=1000)

# Get best solution
best_sample = sampleset.first.sample

# Decode solution
positions, b_matrix = protein_qubo.decode_solution(best_sample)

# Evaluate energy
total_E, breakdown, contacts = evaluator.evaluate(b_matrix, verbose=True)
""")

print(f"\nQUBO statistics:")
print(f"  Variables: {N*N} binary variables")
print(f"  QUBO terms: {len(qubo)}")
print(f"  Problem size suitable for: D-Wave Advantage (5000+ qubits)")

# %% [markdown]
# # 10. Summary
# 
# This notebook demonstrated:
# 
# 1. **Problem Formulation**: Converting lattice protein folding to QUBO
# 2. **PyQUBO Implementation**: Using PyQUBO to build the model programmatically
# 3. **Energy Evaluation**: Analyzing any configuration's energy breakdown
# 4. **Visualization**: Viewing protein configurations on the lattice
# 5. **D-Wave Integration**: Exporting QUBO for quantum annealing
# 
# ## Next Steps:
# 
# - Submit to D-Wave's hybrid solver or quantum annealer
# - Implement classical simulated annealing for comparison
# - Scale up to larger proteins (N=48 as in the paper)
# - Experiment with different lattice topologies (3D cubic lattices)
# - Use real Miyazawa-Jernigan interaction parameters

print("\n" + "="*60)
print("NOTEBOOK COMPLETE")
print("="*60)
