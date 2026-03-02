#!/usr/bin/env python3
"""
entropy_area.py

Count admissible motifs intersecting spheres of various radii.
Assumes motifs.json contains all translated embeddings (or we generate translations).
Plots log(count) vs log(area) to extract exponent and α.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from collections import deque
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paper0'))
from prism_violation import build_hcp_patch

# Parameters
PATCH_RADIUS = 20
PATCH_LAYERS = 30
MOTIF_JSON = 'supplements/motifs.json'   # adjust path
TRANSLATE_MOTIFS = True                   # set False if json already contains all embeddings
RADIUS_STEP = 2
MIN_RADIUS = 4

def load_motif_shapes(json_file):
    """Return list of motif shapes (as frozenset of relative coordinates)."""
    with open(json_file, 'r') as f:
        cert = json.load(f)
    shapes = []
    for m in cert['motifs']:
        # representative_embedding nodes are absolute coordinates; we need relative to centre
        nodes = [tuple(v) for v in m['representative_embedding']['nodes']]
        # Convert to relative by subtracting the first node (or compute min)
        # A simple translation invariant: store as tuple of differences from the min coordinate
        min_coord = tuple(min(c[i] for c in nodes) for i in range(3))
        rel = frozenset(tuple(c[i] - min_coord[i] for i in range(3)) for c in nodes)
        shapes.append(rel)
    return shapes

def generate_all_motifs(shapes, interior_nodes, adj):
    """
    Generate all translated instances of each shape that fit entirely within interior.
    Returns list of motif node sets (as frozensets of absolute coordinates).
    """
    motifs = []
    # For speed, we can precompute a set of interior nodes
    interior_set = set(interior_nodes)
    for shape in shapes:
        # For each interior node as potential origin, translate shape
        for origin in interior_nodes:
            translated = frozenset(tuple(origin[i] + d[i] for i in range(3)) for d in shape)
            if all(v in interior_set for v in translated):
                motifs.append(translated)
    return motifs

def sphere_boundary(adj, centre, radius):
    """Return set of nodes at graph distance exactly radius from centre."""
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
    return {v for v, d in dist.items() if d == radius}

def main():
    print("Building large HCP patch...")
    adj = build_hcp_patch(R=PATCH_RADIUS, L=PATCH_LAYERS)
    interior = [v for v in adj if len(adj[v]) == 12]
    centre = interior[len(interior)//2]
    print(f"Centre: {centre}")

    # Load motif shapes
    print("Loading motif shapes...")
    shapes = load_motif_shapes(MOTIF_JSON)
    print(f"Loaded {len(shapes)} shapes")

    # Generate all translated motifs
    if TRANSLATE_MOTIFS:
        print("Generating all motif placements...")
        motifs = generate_all_motifs(shapes, interior, adj)
        print(f"Generated {len(motifs)} motif instances")
    else:
        # Assume motifs.json already contains absolute embeddings
        with open(MOTIF_JSON, 'r') as f:
            cert = json.load(f)
        motifs = [frozenset(tuple(v) for v in m['representative_embedding']['nodes'])
                  for m in cert['motifs']]
        print(f"Loaded {len(motifs)} motifs (assumed full embeddings)")

    # Precompute distances from centre for all nodes
    print("Computing distances from centre...")
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
    max_r = max(dist.values()) - 2   # stay away from boundary
    radii = list(range(MIN_RADIUS, max_r, RADIUS_STEP))

    counts = []
    areas = []
    for r in radii:
        boundary = sphere_boundary(adj, centre, r)
        cnt = 0
        for m in motifs:
            # Check if motif straddles the sphere: some nodes inside, some outside
            inside = any(dist.get(v, r+1) <= r for v in m)
            outside = any(dist.get(v, r+1) > r for v in m)
            if inside and outside:
                cnt += 1
        counts.append(cnt)
        areas.append(len(boundary))
        print(f"r={r}: crossing motifs = {cnt}, boundary size = {len(boundary)}")

    # Fit log(count) vs log(area) for large r
    log_counts = np.log(counts)
    log_areas = np.log(areas)
    fit_start = max(2, len(radii)//2)   # use second half for fit
    coeffs = np.polyfit(log_areas[fit_start:], log_counts[fit_start:], 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    alpha = np.exp(intercept)
    print(f"Fitted slope: {slope:.3f} (expected 1.0)")
    print(f"alpha = {alpha:.3f}")

    # Plot
    plt.figure()
    plt.loglog(areas, counts, 'o-', label='data')
    plt.loglog(areas[fit_start:], np.exp(intercept) * areas[fit_start:]**slope, '--',
               label=f'fit slope {slope:.2f}')
    plt.xlabel('Surface area (boundary node count)')
    plt.ylabel('Number of crossing motifs')
    plt.title('Entropy–area scaling')
    plt.legend()
    plt.savefig('entropy_area.pdf')
    plt.savefig('entropy_area.png')
    print("Saved entropy_area.pdf")

if __name__ == "__main__":
    main()
