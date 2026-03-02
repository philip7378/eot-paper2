#!/usr/bin/env python3
"""
causal_cone.py

Measure the propagation of a local perturbation on the HCP lattice.
Simulate many random realizations and average the radius of the affected region.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paper0'))
from prism_violation import build_hcp_patch, alt_state, count_opposite
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paper1'))
from motif_enumeration import get_admissible_motifs   # we'll need a function to load motifs

# Parameters
PATCH_RADIUS = 10
PATCH_LAYERS = 15
MAX_STEPS = 30
NUM_RUNS = 200
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_motifs():
    """Load admissible motifs from the JSON certificate (Paper 1)."""
    import json
    with open('supplements/motifs.json', 'r') as f:
        cert = json.load(f)
    motifs = []
    for m in cert['motifs']:
        # Representative embedding nodes (list of tuples)
        nodes = [tuple(v) for v in m['representative_embedding']['nodes']]
        motifs.append(nodes)
    return motifs

def propagate(adj, state, centre, motifs, steps):
    """Run one simulation. Returns list of max distances per step."""
    # We need a rule to pick a motif to flip. For simplicity, at each step,
    # choose a random motif from the list and flip it if it lies entirely within the patch.
    current = state.copy()
    flipped = set([centre])
    radius = [0]
    for t in range(steps):
        # Randomly select a motif
        motif_nodes = random.choice(motifs)
        # Check if all nodes are in the graph
        if all(v in adj for v in motif_nodes):
            # Flip the motif
            for v in motif_nodes:
                current[v] = 1 - current[v]
            flipped.update(motif_nodes)
        # Compute max distance from centre of any flipped node
        max_dist = 0
        # We need graph distances – precompute BFS distances from centre? Too expensive.
        # Approximate: use Euclidean distance in embedding? Not good.
        # Instead, we'll compute graph distance using BFS on the small patch.
        # For now, we'll just compute the graph distance for all flipped nodes.
        # This is O(|flipped| * (V+E)) per step – inefficient for large patches.
        # Better: precompute distances from centre once.
        # For the purpose of this script, we'll rely on the Paper 1 version that used a simpler
        # approximate measure. Let's just use the graph distance computed via BFS.
        # We'll precompute distances from centre for all nodes in the patch at start.
        # That requires a global BFS from centre.
        if t == 0:
            # Precompute distances using BFS
            from collections import deque
            dist = {centre: 0}
            q = deque([centre])
            while q:
                v = q.popleft()
                for nb in adj[v]:
                    if nb not in dist:
                        dist[nb] = dist[v] + 1
                        q.append(nb)
        # Now max_dist = max(dist[v] for v in flipped)
        max_dist = max(dist[v] for v in flipped)
        radius.append(max_dist)
    return radius

def main():
    print("Building HCP patch...")
    adj = build_hcp_patch(R=PATCH_RADIUS, L=PATCH_LAYERS)
    # Choose centre as an interior node
    interior = [v for v in adj if len(adj[v]) == 12]
    centre = interior[len(interior)//2]
    print(f"Centre: {centre}")

    # Load admissible motifs
    print("Loading admissible motifs...")
    motifs = load_motifs()
    print(f"Loaded {len(motifs)} motifs")

    all_radii = []
    for run in range(NUM_RUNS):
        if run % 10 == 0:
            print(f"Run {run}/{NUM_RUNS}")
        rad = propagate(adj, alt_state, centre, motifs, MAX_STEPS)
        all_radii.append(rad)

    # Average radius at each step
    max_len = max(len(r) for r in all_radii)
    avg_radius = [np.mean([r[t] for r in all_radii if t < len(r)]) for t in range(max_len)]

    # Fit linear slope (speed) for large t
    t_vals = np.arange(len(avg_radius))
    fit = np.polyfit(t_vals[5:], avg_radius[5:], 1)
    speed = fit[0]
    print(f"Estimated propagation speed: {speed:.3f} nodes/step")

    # Plot
    plt.figure()
    plt.plot(t_vals, avg_radius, 'o-', label='simulation')
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
