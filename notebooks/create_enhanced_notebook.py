#!/usr/bin/env python3
"""
Create enhanced protein folding notebook with QUBO visualization
Run: python create_enhanced_notebook.py
Output: protein_folding_enhanced.ipynb
"""

import json

notebook_json = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lattice Protein Folding with Quantum Annealing\\n",
    "\\n",
    "Complete implementation with QUBO formulation, classical simulated annealing, and quantum annealing integration.\\n",
    "\\n",
    "**Based on:** Irbäck et al., \\"Folding lattice proteins confined on minimal grids using a quantum-inspired encoding\\" *Physical Review E* **112**, 045302 (2025)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Mathematical Formulation\\n",
    "\\n",
    "Binary encoding: $b_{i,n} = 1$ if amino acid $i$ is at site $n$, else $0$\\n",
    "\\n",
    "Total energy:\\n",
    "$$E = E_{\\\\text{MJ}} + \\\\lambda_1 E_1 + \\\\lambda_2 E_2 + \\\\lambda_3 E_3$$\\n",
    "\\n",
    "**Interaction Energy:**\\n",
    "$$E_{\\\\text{MJ}} = -\\\\sum_{|i-j|>1} C(a_i, a_j) \\\\sum_{\\\\langle n,m \\\\rangle} b_{i,n} b_{j,m}$$\\n",
    "\\n",
    "**Constraints:**\\n",
    "$$E_1 = \\\\sum_i \\\\left(\\\\sum_n b_{i,n} - 1\\\\right)^2 \\\\quad \\\\text{(one site per amino acid)}$$\\n",
    "$$E_2 = \\\\frac{1}{2}\\\\sum_n \\\\sum_{i \\\\neq j} b_{i,n} b_{j,n} \\\\quad \\\\text{(self-avoidance)}$$\\n",
    "$$E_3 = \\\\sum_{i=1}^{N-1} \\\\sum_n b_{i,n} \\\\sum_{\\\\|m-n\\\\|>1} b_{i+1,m} \\\\quad \\\\text{(connectivity)}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install if needed: pip install pyqubo numpy matplotlib networkx pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\\n",
    "from pyqubo import Array, Placeholder\\n",
    "import matplotlib.pyplot as plt\\n",
    "import networkx as nx\\n",
    "from typing import List, Dict, Tuple\\n",
    "import time\\n",
    "import pandas as pd\\n",
    "from IPython.display import display, Markdown\\n",
    "\\n",
    "plt.rcParams['figure.figsize'] = (12, 10)\\n",
    "plt.rcParams['font.size'] = 11\\n",
    "print('✓ Imports successful')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. QUBO Formulation Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class LatticeProteinQUBO:\\n",
    "    def __init__(self, chain, adjacency_matrix, interaction_matrix, lambda1=1.0, lambda2=1.0, lambda3=1.0):\\n",
    "        self.chain = chain\\n",
    "        self.N = len(chain)\\n",
    "        self.adj = adjacency_matrix\\n",
    "        self.C = interaction_matrix\\n",
    "        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3\\n",
    "        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1] == self.N\\n",
    "    \\n",
    "    def build_qubo(self):\\n",
    "        N = self.N\\n",
    "        b = Array.create('b', shape=(N, N), vartype='BINARY')\\n",
    "        E_MJ = sum(-self.C.get((self.chain[i], self.chain[j]), 0) * b[i,n] * b[j,m]\\n",
    "                   for i in range(N) for j in range(i+2, N)\\n",
    "                   for n in range(N) for m in range(N) if self.adj[n,m]==1)\\n",
    "        E1 = sum((sum(b[i,n] for n in range(N))-1)**2 for i in range(N))\\n",
    "        E2 = sum(b[i,n]*b[j,n] for n in range(N) for i in range(N) for j in range(i+1,N))\\n",
    "        E3 = sum(b[i,n]*b[i+1,m] for i in range(N-1) for n in range(N)\\n",
    "                for m in range(N) if self.adj[n,m]==0 and n!=m)\\n",
    "        H = E_MJ + Placeholder('lambda1')*E1 + Placeholder('lambda2')*E2 + Placeholder('lambda3')*E3\\n",
    "        return H.compile()\\n",
    "    \\n",
    "    def get_qubo(self, model, feed_dict=None):\\n",
    "        if feed_dict is None:\\n",
    "            feed_dict = {'lambda1': self.lambda1, 'lambda2': self.lambda2, 'lambda3': self.lambda3}\\n",
    "        return model.to_bqm(feed_dict=feed_dict).to_qubo()\\n",
    "    \\n",
    "    def decode_solution(self, sample):\\n",
    "        N = self.N\\n",
    "        b_matrix = np.zeros((N,N), dtype=int)\\n",
    "        positions = []\\n",
    "        for i in range(N):\\n",
    "            for n in range(N):\\n",
    "                if f'b[{i}][{n}]' in sample and sample[f'b[{i}][{n}]']==1:\\n",
    "                    b_matrix[i,n] = 1\\n",
    "                    positions.append(n)\\n",
    "        return positions, b_matrix\\n",
    "\\n",
    "print('✓ LatticeProteinQUBO defined')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Energy Evaluator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class EnergyEvaluator:\\n",
    "    def __init__(self, chain, adjacency_matrix, interaction_matrix, lambda1=1.0, lambda2=1.0, lambda3=1.0):\\n",
    "        self.chain, self.N = chain, len(chain)\\n",
    "        self.adj, self.C = adjacency_matrix, interaction_matrix\\n",
    "        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3\\n",
    "    \\n",
    "    def compute_E_MJ(self, b, verbose=False):\\n",
    "        E, contacts = 0.0, []\\n",
    "        for i in range(self.N):\\n",
    "            for j in range(i+2, self.N):\\n",
    "                c_ij = self.C.get((self.chain[i], self.chain[j]), 0)\\n",
    "                if c_ij != 0:\\n",
    "                    for n in range(self.N):\\n",
    "                        for m in range(self.N):\\n",
    "                            if b[i,n]==1 and b[j,m]==1 and self.adj[n,m]==1:\\n",
    "                                E += c_ij\\n",
    "                                contacts.append((i,j,self.chain[i],self.chain[j],n,m,c_ij))\\n",
    "        return -E, contacts\\n",
    "    \\n",
    "    def compute_E1(self, b): return int(np.sum((np.sum(b, axis=1)-1)**2))\\n",
    "    def compute_E2(self, b): return int(0.5*sum(np.sum(b[:,n])*(np.sum(b[:,n])-1) for n in range(self.N)))\\n",
    "    def compute_E3(self, b):\\n",
    "        non_adj = 1 - self.adj - np.eye(self.N)\\n",
    "        return int(sum(b[i,:] @ non_adj @ b[i+1,:] for i in range(self.N-1)))\\n",
    "    \\n",
    "    def evaluate(self, b_matrix, verbose=False):\\n",
    "        E_MJ, contacts = self.compute_E_MJ(b_matrix, verbose)\\n",
    "        E1, E2, E3 = self.compute_E1(b_matrix), self.compute_E2(b_matrix), self.compute_E3(b_matrix)\\n",
    "        total = E_MJ + self.lambda1*E1 + self.lambda2*E2 + self.lambda3*E3\\n",
    "        if verbose:\\n",
    "            print(f'E_MJ={E_MJ}, E1={E1}, E2={E2}, E3={E3}')\\n",
    "            print(f'Total={total}, Valid={E1==0 and E2==0 and E3==0}')\\n",
    "        return total, (E_MJ, E1, E2, E3), contacts\\n",
    "\\n",
    "print('✓ EnergyEvaluator defined')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Simulated Annealing Solver"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class SimulatedAnnealing:\\n",
    "    def __init__(self, qubo, offset=0.0):\\n",
    "        self.qubo, self.offset = qubo, offset\\n",
    "        self.variables = list(set([v for edge in qubo.keys() for v in edge]))\\n",
    "        self.n_vars = len(self.variables)\\n",
    "        self.var_to_idx = {v: i for i, v in enumerate(self.variables)}\\n",
    "    \\n",
    "    def energy(self, state):\\n",
    "        E = self.offset\\n",
    "        for (i,j), coeff in self.qubo.items():\\n",
    "            E += coeff * state[self.var_to_idx[i]] * state[self.var_to_idx[j]]\\n",
    "        return E\\n",
    "    \\n",
    "    def solve(self, n_sweeps_per_temp=1000, n_temps=25, beta_start=0.1, beta_end=5.0, n_runs=1, verbose=True):\\n",
    "        betas = np.geomspace(beta_start, beta_end, n_temps)\\n",
    "        best_energy, best_state, all_results = float('inf'), None, []\\n",
    "        \\n",
    "        for run in range(n_runs):\\n",
    "            state = np.random.randint(0, 2, size=self.n_vars)\\n",
    "            current_energy = self.energy(state)\\n",
    "            run_best_energy, run_best_state = current_energy, state.copy()\\n",
    "            \\n",
    "            for beta in betas:\\n",
    "                for _ in range(n_sweeps_per_temp):\\n",
    "                    bit_idx = np.random.randint(0, self.n_vars)\\n",
    "                    state[bit_idx] = 1 - state[bit_idx]\\n",
    "                    new_energy = self.energy(state)\\n",
    "                    if new_energy < current_energy or np.random.random() < np.exp(-beta*(new_energy-current_energy)):\\n",
    "                        current_energy = new_energy\\n",
    "                        if current_energy < run_best_energy:\\n",
    "                            run_best_energy, run_best_state = current_energy, state.copy()\\n",
    "                    else:\\n",
    "                        state[bit_idx] = 1 - state[bit_idx]\\n",
    "            \\n",
    "            sample = {self.variables[i]: int(run_best_state[i]) for i in range(self.n_vars)}\\n",
    "            all_results.append((sample, run_best_energy))\\n",
    "            if run_best_energy < best_energy:\\n",
    "                best_energy, best_state = run_best_energy, run_best_state.copy()\\n",
    "        \\n",
    "        best_sample = {self.variables[i]: int(best_state[i]) for i in range(self.n_vars)}\\n",
    "        if verbose: print(f'Best energy: {best_energy:.2f}')\\n",
    "        return best_sample, best_energy, all_results\\n",
    "\\n",
    "print('✓ SimulatedAnnealing defined')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. QUBO Visualization Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_qubo_algebraic(qubo, offset, max_terms=50):\\n",
    "    \\\"\\\"\\\"Print QUBO in algebraic form\\\"\\\"\\\"\\n",
    "    md = '### QUBO in Algebraic Form\\\\n\\\\n'\\n",
    "    md += f'Constant offset: {offset}\\\\n\\\\n'\\n",
    "    md += '$$E = ' + str(offset) + ' '\\n",
    "    \\n",
    "    terms = []\\n",
    "    for (i, j), coeff in sorted(qubo.items()):\\n",
    "        if abs(coeff) > 1e-10:\\n",
    "            sign = '+' if coeff > 0 else ''\\n",
    "            terms.append(f'{sign}{coeff:.2f} \\\\\\\\cdot {i} \\\\\\\\cdot {j}')\\n",
    "    \\n",
    "    if len(terms) <= max_terms:\\n",
    "        md += ' '.join(terms) + '$$'\\n",
    "    else:\\n",
    "        md += ' '.join(terms[:max_terms]) + f' + \\\\\\\\ldots \\\\\\\\text{{({len(terms)-max_terms} more terms)}}$$'\\n",
    "    \\n",
    "    md += f'\\\\n\\\\nTotal QUBO terms: {len(qubo)}'\\n",
    "    display(Markdown(md))\\n",
    "\\n",
    "def print_qubo_matrix(qubo, offset, max_size=10):\\n",
    "    \\\"\\\"\\\"Print QUBO as a matrix\\\"\\\"\\\"\\n",
    "    variables = sorted(list(set([v for edge in qubo.keys() for v in edge])))\\n",
    "    n = len(variables)\\n",
    "    var_to_idx = {v: i for i, v in enumerate(variables)}\\n",
    "    \\n",
    "    if n > max_size:\\n",
    "        print(f'Matrix too large ({n}x{n}). Showing first {max_size}x{max_size} block:')\\n",
    "        variables = variables[:max_size]\\n",
    "        n = max_size\\n",
    "    \\n",
    "    Q = np.zeros((n, n))\\n",
    "    for (i, j), coeff in qubo.items():\\n",
    "        if i in variables and j in variables:\\n",
    "            idx_i, idx_j = var_to_idx[i], var_to_idx[j]\\n",
    "            if idx_i < n and idx_j < n:\\n",
    "                Q[idx_i, idx_j] = coeff\\n",
    "    \\n",
    "    df = pd.DataFrame(Q, index=variables, columns=variables)\\n",
    "    display(Markdown('### QUBO Matrix Form'))\\n",
    "    display(Markdown(f'Offset: {offset}'))\\n",
    "    display(df.style.format('{:.2f}').background_gradient(cmap='RdBu_r', center=0))\\n",
    "\\n",
    "print('✓ QUBO visualization functions defined')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Protein Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def visualize_lattice(chain, positions, adj, title='', contacts=None):\\n",
    "    G = nx.from_numpy_array(adj)\\n",
    "    pos = nx.spring_layout(G, seed=42, k=0.5)\\n",
    "    fig, ax = plt.subplots(figsize=(12,10))\\n",
    "    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray', ax=ax)\\n",
    "    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=1000, alpha=0.5, ax=ax)\\n",
    "    colors = {'H':'#e74c3c', 'C':'#3498db', 'P':'#2ecc71', 'W':'#9b59b6'}\\n",
    "    if len(positions)==len(chain):\\n",
    "        for i, site in enumerate(positions):\\n",
    "            nx.draw_networkx_nodes(G, pos, nodelist=[site], node_color=colors.get(chain[i],'#f39c12'),\\n",
    "                                  node_size=1200, alpha=0.9, ax=ax, edgecolors='black', linewidths=2)\\n",
    "            ax.text(*pos[site], f'{chain[i]}{i}', fontsize=11, ha='center', va='center', fontweight='bold', color='white')\\n",
    "        nx.draw_networkx_edges(G, pos, edgelist=[(positions[i],positions[i+1]) for i in range(len(chain)-1)],\\n",
    "                              width=4, edge_color='black', ax=ax, alpha=0.7)\\n",
    "        if contacts:\\n",
    "            nx.draw_networkx_edges(G, pos, edgelist=[(n,m) for _,_,_,_,n,m,_ in contacts],\\n",
    "                                  width=3, edge_color='red', style='dashed', alpha=0.6, ax=ax)\\n",
    "    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)\\n",
    "    ax.axis('off')\\n",
    "    return fig\\n",
    "\\n",
    "print('✓ Visualization defined')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\\n",
    "# Example 1: 4-Amino Acid Chain"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Problem Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "chain1 = ['H', 'P', 'C', 'H']\\n",
    "adj1 = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]])\\n",
    "C = {('H','H'):1, ('C','C'):-1, ('H','C'):0, ('C','H'):0, ('H','P'):0, ('P','H'):0, ('C','P'):0, ('P','C'):0, ('P','P'):0}\\n",
    "print(f'Chain: {chain1}, N={len(chain1)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Build QUBO"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "qubo_builder1 = LatticeProteinQUBO(chain1, adj1, C)\\n",
    "model1 = qubo_builder1.build_qubo()\\n",
    "qubo1, offset1 = qubo_builder1.get_qubo(model1)\\n",
    "print(f'QUBO: {len(qubo1)} terms, offset={offset1}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## View QUBO (Algebraic Form)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print_qubo_algebraic(qubo1, offset1, max_terms=30)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## View QUBO (Matrix Form)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print_qubo_matrix(qubo1, offset1, max_size=16)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Solve with Simulated Annealing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sa1 = SimulatedAnnealing(qubo1, offset1)\\n",
    "best_sample1, best_E1, results1 = sa1.solve(n_runs=10, verbose=True)\\n",
    "energies1 = [e for _,e in results1]\\n",
    "print(f'Mean: {np.mean(energies1):.2f}, Std: {np.std(energies1):.2f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluate Solution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluator1 = EnergyEvaluator(chain1, adj1, C)\\n",
    "positions1, b_matrix1 = qubo_builder1.decode_solution(best_sample1)\\n",
    "total1, breakdown1, contacts1 = evaluator1.evaluate(b_matrix1, verbose=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(positions1)==len(chain1):\\n",
    "    fig = visualize_lattice(chain1, positions1, adj1, f'Example 1 Solution (E={total1:.2f})', contacts1)\\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\\n",
    "# Example 2: 6-Amino Acid Chain"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Problem Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "chain2 = ['H', 'H', 'C', 'H', 'P', 'C']\\n",
    "adj2 = np.array([[0, 1, 1, 0, 0, 0],\\n",
    "                 [1, 0, 0, 1, 0, 0],\\n",
    "                 [1, 0, 0, 1, 1, 0],\\n",
    "                 [0, 1, 1, 0, 0, 1],\\n",
    "                 [0, 0, 1, 0, 0, 1],\\n",
    "                 [0, 0, 0, 1, 1, 0]])\\n",
    "print(f'Chain: {chain2}, N={len(chain2)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Build QUBO"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "qubo_builder2 = LatticeProteinQUBO(chain2, adj2, C)\\n",
    "model2 = qubo_builder2.build_qubo()\\n",
    "qubo2, offset2 = qubo_builder2.get_qubo(model2)\\n",
    "print(f'QUBO: {len(qubo2)} terms, offset={offset2}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## View QUBO (Algebraic Form - truncated)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print_qubo_algebraic(qubo2, offset2, max_terms=30)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Solve with Simulated Annealing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sa2 = SimulatedAnnealing(qubo2, offset2)\\n",
    "best_sample2, best_E2, results2 = sa2.solve(n_sweeps_per_temp=2000, n_temps=30, n_runs=10, verbose=True)\\n",
    "energies2 = [e for _,e in results2]\\n",
    "print(f'Mean: {np.mean(energies2):.2f}, Std: {np.std(energies2):.2f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluate Solution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluator2 = EnergyEvaluator(chain2, adj2, C)\\n",
    "positions2, b_matrix2 = qubo_builder2.decode_solution(best_sample2)\\n",
    "total2, breakdown2, contacts2 = evaluator2.evaluate(b_matrix2, verbose=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    if len(positions2)==len(chain2):\\n",
    "    fig = visualize_lattice(chain2, positions2, adj2, f'Example 2 Solution (E={total2:.2f})', contacts2)\\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\\n",
    "# Comparison of Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('='*60)\\n",
    "print('COMPARISON SUMMARY')\\n",
    "print('='*60)\\n",
    "print(f'\\\\nExample 1 (N=4):')\\n",
    "print(f'  Chain: {chain1}')\\n",
    "print(f'  QUBO terms: {len(qubo1)}')\\n",
    "print(f'  Best energy: {best_E1:.2f}')\\n",
    "print(f'  E_MJ: {breakdown1[0]}, Valid: {breakdown1[1]==0 and breakdown1[2]==0 and breakdown1[3]==0}')\\n",
    "print(f'\\\\nExample 2 (N=6):')\\n",
    "print(f'  Chain: {chain2}')\\n",
    "print(f'  QUBO terms: {len(qubo2)}')\\n",
    "print(f'  Best energy: {best_E2:.2f}')\\n",
    "print(f'  E_MJ: {breakdown2[0]}, Valid: {breakdown2[1]==0 and breakdown2[2]==0 and breakdown2[3]==0}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\\n",
    "# D-Wave Integration\\n",
    "\\n",
    "To use with D-Wave quantum annealer:\\n",
    "\\n",
    "```python\\n",
    "from dwave.system import LeapHybridSampler\\n",
    "\\n",
    "# Initialize sampler\\n",
    "sampler = LeapHybridSampler()\\n",
    "\\n",
    "# Submit to D-Wave (Example 1)\\n",
    "sampleset1 = sampler.sample_qubo(qubo1)\\n",
    "dwave_sample1 = sampleset1.first.sample\\n",
    "\\n",
    "# Decode and evaluate\\n",
    "positions, b = qubo_builder1.decode_solution(dwave_sample1)\\n",
    "evaluator1.evaluate(b, verbose=True)\\n",
    "\\n",
    "# For Example 2\\n",
    "sampleset2 = sampler.sample_qubo(qubo2)\\n",
    "dwave_sample2 = sampleset2.first.sample\\n",
    "positions, b = qubo_builder2.decode_solution(dwave_sample2)\\n",
    "evaluator2.evaluate(b, verbose=True)\\n",
    "```\\n",
    "\\n",
    "**Note:** D-Wave's hybrid solver typically solves these in ~10 seconds (as reported in Irbäck et al. 2025 for N=48 systems)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\\n",
    "# Summary\\n",
    "\\n",
    "This notebook demonstrated:\\n",
    "\\n",
    "1. **QUBO Formulation**: Converting protein folding to binary optimization\\n",
    "2. **QUBO Visualization**: Viewing the problem in algebraic and matrix forms\\n",
    "3. **Classical SA**: Solving with simulated annealing as a baseline\\n",
    "4. **Energy Evaluation**: Detailed analysis of solutions\\n",
    "5. **Visualization**: Interactive lattice representations\\n",
    "6. **Multiple Examples**: Comparing different chain sizes\\n",
    "7. **D-Wave Integration**: Ready for quantum annealing\\n",
    "\\n",
    "## Key Insights:\\n",
    "\\n",
    "- **QUBO size scales as O(N²)**: 16 variables for N=4, 36 for N=6\\n",
    "- **Valid solutions** have E₁ = E₂ = E₃ = 0\\n",
    "- **Energy landscape** determined by amino acid interactions (E_MJ)\\n",
    "- **Quantum advantage** expected for larger chains (N > 20)\\n",
    "\\n",
    "## Next Steps:\\n",
    "\\n",
    "- Test with larger chains (N=12, 24, 48)\\n",
    "- Compare SA vs quantum annealing performance\\n",
    "- Experiment with different λ parameters\\n",
    "- Try 3D cubic lattices\\n",
    "- Use real Miyazawa-Jernigan parameters\\n",
    "\\n",
    "## References:\\n",
    "\\n",
    "- Irbäck, Knuthson & Mohanty (2025). Physical Review E 112, 045302\\n",
    "- Miyazawa & Jernigan (1985). Macromolecules 18, 534-552"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "name": "python",
   "mimetype": "text/x-python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''

if __name__ == "__main__":
    print("Creating enhanced protein folding notebook...")
    print("=" * 60)
    
    # Parse and write
    notebook = json.loads(notebook_json)
    
    with open('protein_folding_enhanced.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✓ Created: protein_folding_enhanced.ipynb")
    print()
    print("New Features:")
    print("  ✓ QUBO algebraic form display")
    print("  ✓ QUBO matrix visualization")
    print("  ✓ Two complete examples (N=4 and N=6)")
    print("  ✓ Comparison summary")
    print("  ✓ Enhanced documentation")
    print("  ✓ D-Wave integration guide")
    print("=" * 60)
    print("\nTo use:")
    print("  python create_enhanced_notebook.py")
    print("  jupyter notebook protein_folding_enhanced.ipynb")
 