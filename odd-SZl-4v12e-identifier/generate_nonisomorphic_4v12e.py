from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import importlib.util
import sys
import os

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


# ---------- Dynamic import: SZlOddSolver (odd modulus) ----------

def load_szl_odd_solver_class() -> type:
    here = os.path.dirname(os.path.abspath(__file__))
    solver_path = os.path.join(here, "szl_odd_solver.py")
    spec = importlib.util.spec_from_file_location("szl_odd_solver_module", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load szl_odd_solver.py")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses/typing can resolve annotations.
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "SZlOddSolver"):
        raise ImportError("SZlOddSolver not found in szl_odd_solver.py")
    return module.SZlOddSolver  # type: ignore[return-value]


# ---------- Enumeration + isomorphism reduction ----------

Pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


def degree_from_counts(k: Tuple[int, int, int, int, int, int]) -> Dict[int, int]:
    k12, k13, k14, k23, k24, k34 = k
    return {
        1: k12 + k13 + k14,
        2: k12 + k23 + k24,
        3: k13 + k23 + k34,
        4: k14 + k24 + k34,
    }


def is_connected_from_counts(k: Tuple[int, int, int, int, int, int]) -> bool:
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    for (u, v), w in zip(Pairs, k):
        if w > 0:
            G.add_edge(u, v)
    return nx.is_connected(G)


def canonical_key(k: Tuple[int, int, int, int, int, int]) -> Tuple[int, ...]:
    """Canonicalize a 4-vertex multigraph represented by the 6-tuple of multiplicities.

    We enumerate all 4! vertex permutations and take the lexicographically smallest
    upper-triangular 6-tuple as the canonical key.
    """
    labels = [1, 2, 3, 4]
    # Weight dictionary of the original graph.
    w: Dict[Tuple[int, int], int] = {}
    for (u, v), val in zip(Pairs, k):
        a, b = (u, v) if u < v else (v, u)
        w[(a, b)] = val

    best: Tuple[int, ...] | None = None
    order_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    for perm in itertools.permutations(labels):
        inv = {perm[i]: labels[i] for i in range(4)}  # new -> old
        tup = []
        for (a, b) in order_pairs:  # fixed order in the relabeled graph
            i_old = inv[a]
            j_old = inv[b]
            key = (i_old, j_old) if i_old < j_old else (j_old, i_old)
            tup.append(w[key])
        tup_t = tuple(tup)
        if best is None or tup_t < best:
            best = tup_t
    assert best is not None
    return best


def enumerate_nonisomorphic_graphs() -> List[Tuple[int, int, int, int, int, int]]:
    reps: Dict[Tuple[int, ...], Tuple[int, int, int, int, int, int]] = {}
    # Each multiplicity is in [0,3], total sum is 12, min degree >= 4, and connected.
    for k12 in range(0, 4):
        for k13 in range(0, 4):
            for k14 in range(0, 4):
                for k23 in range(0, 4):
                    for k24 in range(0, 4):
                        for k34 in range(0, 4):
                            k = (k12, k13, k14, k23, k24, k34)
                            if sum(k) != 12:
                                continue
                            deg = degree_from_counts(k)
                            if min(deg.values()) < 4:
                                continue
                            if not is_connected_from_counts(k):
                                continue
                            key = canonical_key(k)
                            reps.setdefault(key, k)
    return list(reps.values())


# ---------- Drawing (grid of small graphs) ----------

def symmetric_rads(m: int, step: float) -> List[float]:
    if m <= 1:
        return [0.0]
    start = -step * (m - 1) / 2.0
    return [start + i * step for i in range(m)]


def draw_case(ax, counts: Tuple[int, int, int, int, int, int], *, hub: int = 1, top: int | None = 2,
              radius: float = 0.8, node_size: int = 280, curve_step: float = 0.18,
              node_color: str = "#66D1B3", edge_color: str = "#5AA9E6") -> None:
    # Nodes and a simple radial layout.
    H = nx.Graph()
    H.add_nodes_from([1, 2, 3, 4])
    for (u, v), w in zip(Pairs, counts):
        if w > 0:
            H.add_edge(u, v)
    pos = {hub: (0.0, 0.0)}
    others = [v for v in [1, 2, 3, 4] if v != hub]
    # Rotate so that 'top' appears at the top (if provided).
    if top in others:
        idx = others.index(top)  # type: ignore[arg-type]
        others = others[idx:] + others[:idx]
    import math
    for i, v in enumerate(others):
        theta = math.pi / 2 + 2 * math.pi * i / len(others)
        pos[v] = (radius * math.cos(theta), radius * math.sin(theta))

    # Nodes and labels.
    nx.draw_networkx_nodes(H, pos, node_color=node_color, node_size=node_size, linewidths=1.0, edgecolors="#2C3E50", ax=ax)
    nx.draw_networkx_labels(H, pos, font_color="#000000", font_size=9, ax=ax)

    # Draw parallel edges as curved arcs (no arrows).
    for (u, v), m in zip(Pairs, counts):
        if m == 0:
            continue
        rads = symmetric_rads(m, curve_step)
        for rad in rads:
            # Odd m includes a straight line; even m uses only curved arcs.
            r = rad
            if m % 2 == 0 and abs(r) < 1e-6:
                r = 0.12
            patch = FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={r}", arrowstyle='-',
                                    color=edge_color, linewidth=1.4, alpha=0.95, zorder=1)
            ax.add_patch(patch)

    # Axes limits and aspect.
    #
    # IMPORTANT: do not use adjustable='datalim' here. With arc patches, Matplotlib may
    # change the data limits differently per subplot, making some graphs look larger/smaller.
    # We instead fix the data limits for every subplot to keep a consistent visual scale.
    pad = 0.22
    lim = radius + pad
    ax.set_autoscale_on(False)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


# ---------- Build a MultiGraph ----------

def build_multigraph(counts: Tuple[int, int, int, int, int, int]) -> nx.MultiGraph:
    Gm = nx.MultiGraph()
    Gm.add_nodes_from([1, 2, 3, 4])
    for (u, v), w in zip(Pairs, counts):
        for _ in range(w):
            Gm.add_edge(u, v)
    return Gm


# ---------- Main ----------

def main():
    l_val = 5
    reps = enumerate_nonisomorphic_graphs()
    print(f"Number of non-isomorphic graphs meeting constraints: {len(reps)}")

    # Draw a grid (5 graphs per row by default).
    cols = 5
    n = len(reps)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    # Flatten axes (robust for ndarray/list/single axes).
    if isinstance(axes, np.ndarray):
        axes_flat = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        axes_flat: List = []
        for row in axes:
            if isinstance(row, (list, tuple, np.ndarray)):
                axes_flat.extend(list(np.array(row).flat))
            else:
                axes_flat.append(row)
    else:
        axes_flat = [axes]

    for i, counts in enumerate(reps):
        ax = axes_flat[i]
        draw_case(ax, counts, hub=1, top=2, radius=0.8)
        ax.set_title(f"#{i+1}", fontsize=9)

    # Hide unused axes.
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    # Save the overview figure to the current working directory (where you run the script).
    overview_filename = f"nonisomorphic_4v12e_SZ_{l_val}_overview.png"
    overview_path = os.path.join(os.getcwd(), overview_filename)
    fig.savefig(overview_path, dpi=300, bbox_inches="tight")
    print(f"Saved overview figure: {overview_path}")
    plt.show()

    # SZ_l testing
    SZlOddSolver = load_szl_odd_solver_class()
    non_sz_cases: List[Tuple[int, Tuple[int, int, int, int, int, int]]] = []
    for i, counts in enumerate(reps):
        Gm = build_multigraph(counts)
        solver = SZlOddSolver(Gm, l_val)
        ok, witness = solver.is_SZl(verbose=False)
        if not ok:
            non_sz_cases.append((i, counts))
            print("\nGraph #{} is NOT SZ_{}:".format(i + 1, l_val))
            print("- Edge multiplicities (sum=12): {}".format(dict(zip(Pairs, counts))))
            deg = degree_from_counts(counts)
            print("- Vertex degrees: {}".format(deg))
            if witness is not None:
                print("- One infeasible beta (vector): {}".format([witness[v] for v in solver.vertices]))

    if not non_sz_cases:
        print("\nAll graphs are SZ_{}.".format(l_val))


if __name__ == "__main__":
    # This script enumerates all non-isomorphic multigraphs meeting constraints,
    # shows them in a grid, and tests each graph for SZ_5.
    main()


