#!/usr/bin/env python3
"""
causal_cone.py

Simulate true successor dynamics on the HCP lattice.
- Use admissible motifs from Paper 1 (motifs.json).
- At each step, randomly pick a motif and check if flipping it preserves admissibility
  in the current configuration. Only accept the flip if it does.
- Measure the maximum graph distance of flipped nodes from the centre over steps.
- Average over many runs to estimate propagation speed.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import json
from collections import deque
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paper0'))
from prism_violation import build_hcp_patch, alt_state, count_opposite

# Parameters
PATCH_RADIUS = 10
PATCH_LAYERS = 20            # larger to avoid boundary effects
MAX_STEPS = 30
NUM_RUNS = 200
SEED = 42
MOTIF_JSON = 'supplements/motifs.json'   # adjust path if needed

random.seed(SEED)
np.random.seed(SEED)

def load_motifs(json_file):
    """Return list of motif node sets (as frozensets) from the certificate."""
    with open(json_file, 'r') as f:
        cert = json.load(f)
    motifs = []
    for m in cert['motifs']:
        nodes = [tuple(v) for v in m['representative_embedding']['nodes']]
        motifs.append(frozenset(nodes))
    return motifs

def is_admissible_flip(adj, state, motif_nodes):
    """
    Check if flipping the given motif preserves admissibility.
    Returns (True, None) if admissible, else (False, (v, pre, post)) for first failing node.
    """
    affected = set(motif_nodes)
    for v in motif_nodes:
        affected.update(adj[v])
    # Only interior nodes (degree 12) matter for equilibrium
    interior = [v for v in affected if len(adj[v]) == 12]
    # Build post‑flip state
    new_state = state.copy()
    for v in motif_nodes:
        new_state[v] = 1 - new_state[v]
    for v in interior:
        pre = count_opposite(adj, state, v)
        post = count_opposite(adj, new_state, v)
        if post != 6:
            return False, (v, pre, post)
    return True, None

def propagate(adj, start_state, centre, motifs, steps):
    """
    Run one simulation starting from start_state, with centre fixed.
    Returns list of max graph distances from centre at each step.
    """
    # Precompute graph distances from centre using BFS
    dist = {centre: 0}
    q = deque([centre])
    while q:
        v = q.popleft()
        d = dist[v]
        for nb in adj[v]:
            if nb not in dist:
                dist[nb] = d+1
                q.append(nb)
    # Make a mutable copy of the state
    current = start_state.copy()
    flipped = set([centre])
    radius = [0]
    for step in range(steps):
        # Randomly permute motifs to try different ones
        random.shuffle(motifs)
        applied = False
        for motif in motifs:
            # Check that all motif nodes are inside the patch (dist contains them)
            if not all(v in dist for v in motif):
                continue
            ok, _ = is_admissible_flip(adj, current, motif)
            if ok:
                # Apply flip
                for v in motif:
                    current[v] = 1 - current[v]
                flipped.update(motif)
                applied = True
                break
        if not applied:
            # No admissible move available; stop
            break
        # Compute current max distance
        max_dist = max(dist[v] for v in flipped)
        radius.append(max_dist)
    return radius

def main():
    print("Building HCP patch...")
    adj = build_hcp_patch(R=PATCH_RADIUS, L=PATCH_LAYERS)
    # Choose centre as an interior node (degree 12)
    interior = [v for v in adj if len(adj[v]) == 12]
    centre = interior[len(interior)//2]
    print(f"Centre: {centre}")

    # Load admissible motifs
    print("Loading admissible motifs...")
    motifs = load_motifs(MOTIF_JSON)
    print(f"Loaded {len(motifs)} motifs")

    # Base state: alternating vacuum
    base_state = {v: alt_state(v) for v in adj}

    all_radii = []
    for run in range(NUM_RUNS):
        if run % 20 == 0:
            print(f"Run {run}/{NUM_RUNS}")
        rad = propagate(adj, base_state, centre, motifs, MAX_STEPS)
        all_radii.append(rad)

    # Determine maximum length and pad with last value
    max_len = max(len(r) for r in all_radii)
    padded = np.array([r + [r[-1]]*(max_len-len(r)) for r in all_radii])
    avg_radius = np.mean(padded, axis=0)

    # Fit linear slope (speed) for steps where radius > 0
    t_vals = np.arange(len(avg_radius))
    # Exclude early steps to avoid initial transients
    fit_start = 5
    fit_end = len(avg_radius)
    if fit_end > fit_start:
        coeffs = np.polyfit(t_vals[fit_start:fit_end], avg_radius[fit_start:fit_end], 1)
        speed = coeffs[0]
        print(f"Estimated propagation speed: {speed:.3f} nodes/step")
    else:
        speed = 0
        print("Not enough steps to fit speed.")

    # Plot
    plt.figure()
    plt.plot(t_vals, avg_radius, 'o-', label='simulation')
    if speed > 0:
        plt.plot(t_vals, speed * t_vals, '--', label=f'fit speed {speed:.3f}')
    plt.xlabel('Step t')
    plt.ylabel('Average radius')
    plt.title('Emergent causal cone')
    plt.legend()
    plt.savefig('causal_cone.pdf')
    plt.savefig('causal_cone.png')
    print("Saved causal_cone.pdf")

if __name__ == "__main__":
    main()
