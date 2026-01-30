from __future__ import annotations

import argparse
import itertools
from typing import Dict, List, Tuple, Optional, TextIO

import importlib.util
import sys
import os

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


# ---------- Dynamic import: SZlOddSolver (odd modulus) ----------

def load_szl_odd_solver_class() -> type:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
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


def enumerate_nonisomorphic_graphs(modulus: int) -> List[Tuple[int, int, int, int, int, int]]:
    """Enumerate 4-vertex multigraphs with 3(l-1) edges (counting multiplicity), where l is odd.
    Constraints: each pair multiplicity in [0, l-2], total edges = 3(l-1), min degree >= l-1, connected.
    """
    if modulus <= 0 or modulus % 2 == 0:
        raise ValueError("modulus must be a positive odd integer.")
    total_edges = 3 * (modulus - 1)
    min_degree = modulus - 1
    max_multiplicity = modulus - 2  # per pair: 0 .. l-2
    reps: Dict[Tuple[int, ...], Tuple[int, int, int, int, int, int]] = {}
    r = range(0, max_multiplicity + 1)  # 0 .. l-2
    for k12 in r:
        for k13 in r:
            for k14 in r:
                for k23 in r:
                    for k24 in r:
                        for k34 in r:
                            k = (k12, k13, k14, k23, k24, k34)
                            if sum(k) != total_edges:
                                continue
                            deg = degree_from_counts(k)
                            if min(deg.values()) < min_degree:
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


def log_print(msg: str, log_file: Optional[TextIO]) -> None:
    """Print to stdout and optionally write the same line to log_file."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate 4-vertex, 3(l-1)-edge non-isomorphic multigraphs (min degree >= l-1, max multiplicity l-2) and test each for SZ_l."
    )
    parser.add_argument(
        "--modulus", "-l", type=int, default=5,
        help="Odd modulus l (default: 5). Total edges = 3(l-1), min degree >= l-1, max multiplicity per pair = l-2."
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show(); only save the overview figure."
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Do not write run output to a results txt file."
    )
    args = parser.parse_args()
    l_val = args.modulus
    if l_val <= 0 or l_val % 2 == 0:
        raise ValueError("--modulus must be a positive odd integer.")
    total_edges = 3 * (l_val - 1)

    # Open results file (same directory as overview PNG: current working directory)
    log_file: Optional[TextIO] = None
    if not args.no_log:
        results_filename = "nonisomorphic_4v{}e_l{}_results.txt".format(total_edges, l_val)
        results_path = os.path.join(os.getcwd(), results_filename)
        try:
            log_file = open(results_path, "w", encoding="utf-8")
        except OSError:
            log_file = None  # fallback to console only

    def out(msg: str) -> None:
        log_print(msg, log_file)

    reps = enumerate_nonisomorphic_graphs(l_val)
    out("Modulus l = {}, total edges = {}, min degree >= {}, max multiplicity per pair = {}".format(
        l_val, total_edges, l_val - 1, l_val - 2))
    out("Number of non-isomorphic graphs meeting constraints: {}".format(len(reps)))

    # Draw a grid (5 graphs per row by default).
    cols = 5
    n = len(reps)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    # Flatten axes (robust for ndarray/list/single axes).
    if isinstance(axes, np.ndarray):
        axes_flat = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        axes_flat = []
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
    overview_filename = "nonisomorphic_4v{}e_l{}_overview.png".format(total_edges, l_val)
    overview_path = os.path.join(os.getcwd(), overview_filename)
    fig.savefig(overview_path, dpi=300, bbox_inches="tight")
    out("Saved overview figure: {}".format(overview_path))
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    # SZ_l testing: collect and print all infeasible betas for each non-SZ_l graph
    SZlOddSolver = load_szl_odd_solver_class()
    non_sz_cases: List[Tuple[int, Tuple[int, int, int, int, int, int]]] = []
    for i, counts in enumerate(reps):
        Gm = build_multigraph(counts)
        solver = SZlOddSolver(Gm, l_val)
        infeasible_betas = solver.get_all_infeasible_betas(verbose=False)
        if infeasible_betas:
            non_sz_cases.append((i, counts))
            out("")
            out("Graph #{} is NOT SZ_{}:".format(i + 1, l_val))
            out("- Edge multiplicities (sum={}): {}".format(total_edges, dict(zip(Pairs, counts))))
            deg = degree_from_counts(counts)
            out("- Vertex degrees: {}".format(deg))
            out("- All infeasible betas (vectors in vertex order {}):".format(solver.vertices))
            for idx, beta in enumerate(infeasible_betas):
                vec = [beta[v] for v in solver.vertices]
                out("  [{}] {}".format(idx + 1, vec))
            out("  Total: {} infeasible beta(s).".format(len(infeasible_betas)))

    if not non_sz_cases:
        out("")
        out("All graphs are SZ_{}.".format(l_val))

    if log_file is not None:
        out("")
        out("Results written to: {}".format(results_path))
        log_file.close()

if __name__ == "__main__":
    main()
