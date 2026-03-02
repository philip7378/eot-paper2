# Paper 2 Code: Emergent Geometry and Lorentz Symmetry

This directory contains Python scripts to reproduce the numerical experiments and figures for Paper 2 of Equilibrium Ordering Theory (EOT).

## Requirements

- Python 3.8+
- Packages: `numpy`, `scipy`, `networkx`, `matplotlib`
- The code from Paper 0 (HCP builder) and Paper 1 (motif enumeration) is used; ensure the paths are set correctly.

## Scripts

- `spectral_convergence.py` – Computes low‑lying eigenvalues of the graph Laplacian on HCP patches of increasing size. Plots convergence of λ₁ and saves eigenvalues.
- `causal_cone.py` – Simulates many random realizations of successor dynamics, measures the average radius of the perturbed region vs. step count, and estimates the propagation speed.
- `entropy_area.py` – Loads the motif certificate from Paper 1 (`motifs.json`), counts motifs intersecting spheres of various radii, and verifies the linear scaling with surface area (entropy–area law).
- `plot_figures.py` – (optional) Combines data from the above scripts to generate the final figures for the paper.

## Usage

Run each script from the command line. For example:
