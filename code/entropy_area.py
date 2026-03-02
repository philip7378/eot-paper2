#!/usr/bin/env python3
"""
entropy_area.py

Count admissible motifs intersecting spheres of various radii.
Plots log count vs surface area to extract α.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paper0'))
from prism_violation import build_hcp_patch

# Parameters
PATCH_RADIUS = 20
PATCH_LAYERS = 25
MOTIF_JSON = 'supplements/motifs.json'   # path to motif certificate

def load_motif_embeddings(json_file):
    """Return list of motif node sets (each as set of tuples)."""
    with open(json_file, 'r') as f:
        cert = json.load(f)
    motifs = []
    for m in cert['motifs']:
        nodes = [tuple(v) for v in m['representative_embedding']['nodes']]
        motifs.append(set(nodes))
    return motifs

def sphere_boundary(adj, centre, radius):
    """Return the set of nodes at distance exactly radius from centre (BFS shell)."""
    from collections import deque
    dist = {centre: 0}
    q = deque([centre])
    while q:
        v = q.popleft()
        d = dist[v]
        if d == radius:
            continue
        for nb in adj[v]:
            if nb not in dist:
                dist[nb] = d+1
                q.append(nb)
    # Return nodes at distance exactly radius
    return {v for v, d in dist.items() if d == radius}

def main():
    print("Building large HCP patch...")
    adj = build_hcp_patch(R=PATCH_RADIUS, L=PATCH_LAYERS)
    interior = [v for v in adj if len(adj[v]) == 12]
    centre = interior[len(interior)//2]   # pick an interior centre
    print(f"Centre: {centre}")

    print("Loading motif embeddings...")
    motifs = load_motif_embeddings(MOTIF_JSON)
    print(f"Loaded {len(motifs)} motifs")

    # Choose radii to consider (up to half the patch size to avoid boundary)
    max_r = min(PATCH_RADIUS, PATCH_LAYERS//2) - 2
    radii = range(3, max_r+1, 2)   # start from small radius to avoid core
    counts = []

    for r in radii:
        # Get the boundary at distance r (nodes exactly at distance r)
        boundary = sphere_boundary(adj, centre, r)
        # Count motifs that intersect both the ball of radius r and its complement.
        # That is, motifs with at least one node inside the ball and one outside.
        # The ball of radius r is all nodes with distance <= r.
        # We'll compute the distance from centre for each motif node (costly).
        # Precompute distances from centre once.
        if r == radii[0]:
            # compute distances from centre for all nodes
            from collections import deque
            dist = {centre: 0}
            q = deque([centre])
            while q:
                v = q.popleft()
                d = dist[v]
                for nb in adj[v]:
                    if nb not in dist:
                        dist[nb] = d+1
                        q.append(nb)
            # dist now contains distances for all nodes
        # Count crossing motifs
        cnt = 0
        for m in motifs:
            # Check if any node in m is inside (dist <= r) and any node outside (dist > r)
            inside = any(dist.get(v, r+1) <= r for v in m if v in dist)
            outside = any(dist.get(v, r+1) > r for v in m if v in dist)
            if inside and outside:
                cnt += 1
        counts.append(cnt)
        print(f"r={r}: crossing motifs = {cnt}")

    # Area: number of nodes on the boundary (approximate area)
    areas = [len(sphere_boundary(adj, centre, r)) for r in radii]

    # Fit log(count) vs log(area) for large r (should be slope 1)
    log_counts = np.log(counts)
    log_areas = np.log(areas)
    fit = np.polyfit(log_areas[2:], log_counts[2:], 1)
    alpha = np.exp(fit[1])   # intercept
    slope = fit[0]
    print(f"Fitted slope: {slope:.3f} (expected 1.0)")
    print(f"alpha = {alpha:.3f}")

    # Plot
    plt.figure()
    plt.loglog(areas, counts, 'o-', label='data')
    plt.loglog(areas, alpha * areas**slope, '--', label=f'fit slope {slope:.2f}')
    plt.xlabel('Surface area (boundary node count)')
    plt.ylabel('Number of crossing motifs')
    plt.title('Entropy–area scaling')
    plt.legend()
    plt.savefig('entropy_area.pdf')
    plt.savefig('entropy_area.png')
    print("Saved entropy_area.pdf")

if __name__ == "__main__":
    main()
