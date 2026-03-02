# Paper 2 Code: Emergent Geometry and Lorentz Symmetry

This directory contains Python scripts to reproduce the numerical experiments and figures for Paper 2 of Equilibrium Ordering Theory (EOT).

## Requirements

- Python 3.8+
- Packages: `numpy`, `scipy`, `matplotlib`, `networkx`
- The code from Paper 0 (`prism_violation.py`) and the motif certificate from Paper 1 (`motifs.json`) must be accessible. Adjust paths in the scripts accordingly.

## Scripts

- `spectral_convergence.py` – Computes low‑lying eigenvalues of the graph Laplacian on HCP patches of increasing size. Uses sparse matrices and scales the domain to be roughly cubic. Plots convergence of λ₁ vs. 1/L² and fits the slope to verify λ₁ ∝ L⁻².
- `causal_cone.py` – Simulates true successor dynamics: randomly picks admissible motifs from the Paper 1 certificate, checks admissibility in the current configuration, and only flips if the motif preserves equilibrium. Measures the average radius of the affected region over many runs and estimates the propagation speed.
- `entropy_area.py` – Loads motif shapes from the certificate, generates all translated instances (or assumes full embeddings), counts motifs intersecting spheres of various radii, and fits log(count) vs. log(area) to verify the entropy–area law. Uses BFS distance precomputation for efficiency.

## Usage

Run each script from the command line. For example:
