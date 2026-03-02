#!/usr/bin/env python3
"""
spectral_convergence.py

Compute low‑lying eigenvalues of the graph Laplacian on HCP patches of increasing size.
Plots convergence to continuum eigenvalues.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paper0'))
from prism_violation import build_hcp_patch

# Parameters
patch_sizes = [6, 10, 14, 18, 22]   # approximate linear dimension (R)
layers = 8                            # enough to avoid boundary effects
num_eigs = 10                          # number of eigenvalues to compute

def graph_laplacian(adj):
    """Return the graph Laplacian matrix (combinatorial) for the adjacency dict."""
    nodes = list(adj.keys())
    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    # Build adjacency matrix
    A = np.zeros((n, n))
    for v, nbrs in adj.items():
        i = node_index[v]
        for w in nbrs:
            j = node_index[w]
            A[i, j] = 1
    # Degree matrix
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L, node_index

def compute_eigenvalues(adj, k):
    """Compute the k smallest eigenvalues of the graph Laplacian."""
    L, _ = graph_laplacian(adj)
    # Use sparse eigensolver for efficiency
    eigvals, _ = eigsh(L, k=k, which='SM', return_eigenvectors=False)
    return np.sort(eigvals)

def main():
    results = {}
    for R in patch_sizes:
        print(f"Building patch with R={R}, L={layers}...")
        adj = build_hcp_patch(R=R, L=layers)
        # Extract interior nodes only to avoid boundary effects
        interior = [v for v in adj if len(adj[v]) == 12]
        # Build subgraph on interior
        interior_adj = {v: [w for w in adj[v] if w in interior] for v in interior}
        # Ensure connectivity (should be connected)
        print(f"  Interior nodes: {len(interior)}")
        eig = compute_eigenvalues(interior_adj, num_eigs)
        results[R] = eig
        print(f"  First {num_eigs} eigenvalues: {eig}")

    # Save results
    np.savez('eigenvalues.npz', **results)

    # Plot convergence of first non‑zero eigenvalue (λ₁)
    plt.figure()
    sizes = np.array([R for R in patch_sizes])
    lambda1 = [results[R][1] for R in patch_sizes]  # index 0 is zero
    plt.loglog(sizes, lambda1, 'o-', label='λ₁')
    # Expected continuum scaling: λ₁ ~ 1/L² for a box, but here L is linear size.
    # For a cubic box of side L, first eigenvalue scales as 1/L².
    # We'll plot a reference line.
    fit = np.polyfit(np.log(sizes[2:]), np.log(lambda1[2:]), 1)
    plt.loglog(sizes, np.exp(fit[1]) * sizes**fit[0], '--',
               label=f'fit slope {fit[0]:.2f}')
    plt.xlabel('Patch linear size R')
    plt.ylabel('First non‑zero eigenvalue λ₁')
    plt.legend()
    plt.title('Spectral convergence of graph Laplacian')
    plt.savefig('spectral_convergence.pdf')
    plt.savefig('spectral_convergence.png')
    print("Saved spectral_convergence.pdf")

if __name__ == "__main__":
    main()
