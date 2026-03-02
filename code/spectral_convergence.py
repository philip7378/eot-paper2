#!/usr/bin/env python3
"""
spectral_convergence.py

Compute low‑lying eigenvalues of the graph Laplacian on HCP patches of increasing size.
Uses sparse matrices to handle large patches. Scales the patch to be roughly cubic.
Plots convergence of λ₁ vs. 1/L² and fits slope to verify λ₁ ∝ L⁻².
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paper0'))
from prism_violation import build_hcp_patch

# Parameters
R_values = [6, 10, 14, 18, 22]       # horizontal radius
layers_factor = 2                      # ensure cubic domain: L ≈ 2R
num_eigs = 10                          # number of eigenvalues to compute

def graph_laplacian_sparse(adj):
    """Return the graph Laplacian as a sparse CSR matrix."""
    nodes = list(adj.keys())
    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    # Build adjacency matrix in LIL format for efficient construction
    A = lil_matrix((n, n), dtype=np.float64)
    for v, nbrs in adj.items():
        i = node_index[v]
        for w in nbrs:
            j = node_index[w]
            A[i, j] = 1.0
    # Convert to CSR for arithmetic
    A = A.tocsr()
    # Degree matrix as diagonal (sparse)
    d = A.sum(axis=1).A1
    D = csr_matrix((d, (range(n), range(n))), shape=(n, n))
    L = D - A
    return L, node_index

def compute_eigenvalues(adj, k):
    """Compute the k smallest eigenvalues of the graph Laplacian."""
    L, _ = graph_laplacian_sparse(adj)
    # Use shift‑invert mode for stability and speed
    # sigma=0 looks for eigenvalues near zero (the smallest non‑zero ones)
    eigvals, _ = eigsh(L, k=k, sigma=0, which='LM', return_eigenvectors=False)
    return np.sort(eigvals)

def main():
    results = {}
    for R in R_values:
        L_layers = layers_factor * R
        print(f"Building patch with R={R}, L={L_layers}...")
        adj = build_hcp_patch(R=R, L=L_layers)
        # Extract interior nodes (degree 12)
        interior = [v for v in adj if len(adj[v]) == 12]
        print(f"  Interior nodes: {len(interior)}")
        # Build subgraph on interior (must be connected; if not, take the largest component)
        interior_adj = {v: [w for w in adj[v] if w in interior] for v in interior}
        # Optionally: ensure connectivity via largest connected component
        # (implementation omitted for brevity; assume interior is connected for HCP)
        eig = compute_eigenvalues(interior_adj, num_eigs)
        results[R] = eig
        print(f"  First {num_eigs} eigenvalues: {eig}")

    # Save results
    np.savez('eigenvalues.npz', **results)

    # Plot convergence of first non‑zero eigenvalue λ₁ (index 1, since index 0 is zero)
    sizes = np.array([R for R in R_values])
    lambda1 = [results[R][1] for R in R_values]

    # Fit λ₁ = a * L⁻²  => log(λ₁) = log(a) -2 log(L)
    # Here L is the linear size (≈ 2R)
    L_vals = 2 * sizes
    logL = np.log(L_vals)
    logλ = np.log(lambda1)
    # Fit for larger sizes (exclude smallest to avoid strong finite‑size effects)
    fit_range = slice(2, None)  # skip first two points
    coeffs = np.polyfit(logL[fit_range], logλ[fit_range], 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    print(f"Fitted slope: {slope:.3f} (expected -2.0)")

    plt.figure()
    plt.loglog(L_vals, lambda1, 'o-', label='data')
    plt.loglog(L_vals[fit_range], np.exp(intercept) * L_vals[fit_range]**slope, '--',
               label=f'fit slope {slope:.2f}')
    plt.xlabel('Linear size L (≈ 2R)')
    plt.ylabel('First non‑zero eigenvalue λ₁')
    plt.title('Spectral convergence of graph Laplacian')
    plt.legend()
    plt.savefig('spectral_convergence.pdf')
    plt.savefig('spectral_convergence.png')
    print("Saved spectral_convergence.pdf")

if __name__ == "__main__":
    main()
