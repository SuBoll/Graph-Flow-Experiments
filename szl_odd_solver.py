from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import networkx as nx


@dataclass(frozen=True)
class EdgeBundle:
    """An undirected vertex pair (no self-loops) with multiplicity, stored as (u, v, k) with u < v."""
    u: int
    v: int
    k: int


@dataclass
class OddOrientationSolution:
    """A beta-orientation solution for odd modulus l.

    - y_by_pair[(u,v)]: among the k parallel edges between u and v (u < v),
      y edges are oriented u -> v.
    - out_minus_in[v]: integer (outdegree - indegree) at vertex v (not reduced mod l).
    - beta: the given beta (values 0..l-1, with sum(beta) ≡ 0 mod l).
    - directions: per-edge orientation list, each item is (tail, head).
    """
    modulus: int
    vertices: List[int]
    edge_bundles: List[EdgeBundle]
    y_by_pair: Dict[Tuple[int, int], int]
    out_minus_in: Dict[int, int]
    beta: Dict[int, int]
    directions: List[Tuple[int, int]]

    def pretty_print(self) -> None:
        mod = self.modulus
        print("—— One beta-orientation solution (odd l) ——")
        print(f"l={mod}")
        print("beta (vector):", [self.beta[v] for v in self.vertices])
        print("Vertex out-in and check (mod l):")
        for v in self.vertices:
            val = self.out_minus_in[v]
            print(f"  v={v}: out-in={val}, mod {mod} = {val % mod}")
        print("Pairs (u,v), k, y(u->v), contribution to u (2y-k):")
        for eb in self.edge_bundles:
            y = self.y_by_pair[(eb.u, eb.v)]
            contrib = 2 * y - eb.k
            print(f"  ({eb.u},{eb.v}), k={eb.k}, y={y}, 2y-k={contrib}")
        print("Per-edge directions (first few):")
        for i, (a, b) in enumerate(self.directions[:40], 1):
            print(f"  e{i}: {a}->{b}")
        if len(self.directions) > 40:
            print(f"  ... total edges (counting multiplicity): {len(self.directions)}")


class SZlOddSolver:
    """SZ_l decision and solving for odd modulus l only (beta ∈ Z_l, no 2l / parity constraints).

    Given a connected undirected multigraph (no self-loops). For each unordered pair {u,v}
    with multiplicity k (stored as u < v). If among those k parallel edges there are y edges
    oriented u -> v, then:
      - contribution at u to (out-in) is y - (k-y) = 2y - k
      - contribution at v is -(2y - k)
    Thus the contribution set per pair is {k, k-2, ..., -k} (and the opposite at the other end).

    SZ_l: for every beta: V -> Z_l (values 0..l-1, sum(beta) ≡ 0 mod l),
    there exists a beta-orientation.
    """

    def __init__(self, multigraph: nx.MultiGraph, modulus: int):
        if modulus <= 0 or modulus % 2 == 0:
            raise ValueError("This solver supports odd modulus l only (positive odd integer).")
        if any(u == v for u, v in multigraph.edges()):
            raise ValueError("Self-loops are not allowed.")
        if not nx.is_connected(nx.Graph(multigraph)):
            raise ValueError("The graph must be connected.")

        self.Gm: nx.MultiGraph = multigraph
        self.l: int = modulus

        self.vertices: List[int] = sorted(self.Gm.nodes())
        self.index_of_vertex: Dict[int, int] = {v: i for i, v in enumerate(self.vertices)}
        self.edge_bundles: List[EdgeBundle] = self._collect_edge_bundles()
        self.sign_by_vertex: List[List[Tuple[int, int]]] = self._build_signs()
        self.deg: Dict[int, int] = self._compute_degrees()

        # Define C_v = sum_e sign(v,e) * (-k_e), so that:
        #   C_v + 2 * sum_e sign(v,e) * y_e ≡ beta(v) (mod l)
        self.C_vec: List[int] = [
            sum(sign * (-self.edge_bundles[eidx].k) for eidx, sign in self.sign_by_vertex[v_idx])
            for v_idx in range(len(self.vertices))
        ]

        # 2 is invertible in Z_l since l is odd.
        self.inv2: int = pow(2, -1, self.l)

    def _collect_edge_bundles(self) -> List[EdgeBundle]:
        count: Dict[Tuple[int, int], int] = {}
        for u, v in self.Gm.edges():
            a, b = (u, v) if u < v else (v, u)
            count[(a, b)] = count.get((a, b), 0) + 1
        return [EdgeBundle(u=a, v=b, k=k) for (a, b), k in sorted(count.items())]

    def _build_signs(self) -> List[List[Tuple[int, int]]]:
        n = len(self.vertices)
        sign_by_vertex: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        for eidx, eb in enumerate(self.edge_bundles):
            u_idx = self.index_of_vertex[eb.u]
            v_idx = self.index_of_vertex[eb.v]
            sign_by_vertex[u_idx].append((eidx, +1))
            sign_by_vertex[v_idx].append((eidx, -1))
        return sign_by_vertex

    def _compute_degrees(self) -> Dict[int, int]:
        deg: Dict[int, int] = {v: 0 for v in self.vertices}
        for eb in self.edge_bundles:
            deg[eb.u] += eb.k
            deg[eb.v] += eb.k
        return deg

    # ---------- beta enumeration ----------

    def enumerate_betas(self) -> Iterable[Dict[int, int]]:
        """Enumerate all beta: V->Z_l (0..l-1) with sum(beta) ≡ 0 (mod l)."""
        n = len(self.vertices)
        mod = self.l
        # Enumerate the first n-1 values; the last one is determined by sum(beta) ≡ 0.
        for values in itertools.product(range(mod), repeat=n - 1):
            s = sum(values) % mod
            last = (-s) % mod
            beta = {v: val for v, val in zip(self.vertices[:-1], values)}
            beta[self.vertices[-1]] = last
            yield beta

    # ---------- solving ----------

    def solve_for_beta(self, beta: Dict[int, int]) -> Tuple[bool, Optional[OddOrientationSolution]]:
        """Given beta (values 0..l-1 with sum ≡ 0), find a beta-orientation."""
        if set(beta.keys()) != set(self.vertices):
            raise ValueError("beta's vertex set does not match the graph's vertex set.")
        mod = self.l
        if any((beta[v] < 0 or beta[v] >= mod) for v in self.vertices):
            return False, None
        if sum(beta.values()) % mod != 0:
            return False, None

        # gamma(v) = inv2*(beta(v) - C_v) mod l
        gamma: List[int] = []
        for idx, v in enumerate(self.vertices):
            gamma.append((self.inv2 * ((beta[v] - self.C_vec[idx]) % mod)) % mod)

        # Variables: y_e ∈ [0..k_e]
        domains: List[List[int]] = [list(range(0, eb.k + 1)) for eb in self.edge_bundles]
        order = sorted(range(len(self.edge_bundles)), key=lambda eidx: len(domains[eidx]))

        y_sol: List[Optional[int]] = [None] * len(self.edge_bundles)
        partial_sum = [0] * len(self.vertices)  # partial integer sums of sum(sign*y) at each vertex

        def remaining_range_for_vertex(v_idx: int, next_pos: int) -> Tuple[int, int]:
            L = 0
            U = 0
            for i_pos in range(next_pos, len(order)):
                eidx = order[i_pos]
                eb = self.edge_bundles[eidx]
                # For a given vertex, sign*y ranges over [0..k] (sign=+1) or [-k..0] (sign=-1).
                for eidx2, sign in self.sign_by_vertex[v_idx]:
                    if eidx2 == eidx:
                        if sign == +1:
                            U += eb.k
                        else:
                            L -= eb.k
                        break
            return L, U

        def dfs(pos: int) -> bool:
            if pos == len(order):
                # Check congruence constraints.
                for v_idx in range(len(self.vertices)):
                    if (partial_sum[v_idx] - gamma[v_idx]) % mod != 0:
                        return False
                return True

            eidx = order[pos]
            eb = self.edge_bundles[eidx]
            # Endpoints
            u_idx = self.index_of_vertex[eb.u]
            v_idx = self.index_of_vertex[eb.v]

            for y in domains[eidx]:
                # Apply
                partial_sum[u_idx] += y
                partial_sum[v_idx] -= y

                # Prune: for each vertex, check if completion can still hit the required residue mod l.
                pruned = False
                next_pos = pos + 1
                for vv in range(len(self.vertices)):
                    L, U = remaining_range_for_vertex(vv, next_pos)
                    need = (gamma[vv] - (partial_sum[vv] % mod)) % mod
                    left = L - need
                    right = U - need
                    if right - left + 1 < mod:
                        qmin = (left + mod - 1) // mod
                        qmax = right // mod
                        if qmin > qmax:
                            pruned = True
                            break

                if not pruned and dfs(next_pos):
                    y_sol[eidx] = y
                    return True

                # Roll back
                partial_sum[u_idx] -= y
                partial_sum[v_idx] += y

            return False

        ok = dfs(0)
        if not ok:
            return False, None

        # Assemble solution
        y_by_pair: Dict[Tuple[int, int], int] = {}
        out_minus_in: Dict[int, int] = {v: 0 for v in self.vertices}
        directions: List[Tuple[int, int]] = []

        for eidx, eb in enumerate(self.edge_bundles):
            y = y_sol[eidx]
            assert y is not None
            y_by_pair[(eb.u, eb.v)] = y
            contrib = 2 * y - eb.k
            out_minus_in[eb.u] += contrib
            out_minus_in[eb.v] -= contrib
            # Per-edge directions: first y edges are eb.u->eb.v, remaining eb.k-y are reversed.
            directions.extend([(eb.u, eb.v)] * y)
            directions.extend([(eb.v, eb.u)] * (eb.k - y))

        # Verify (mod l)
        for v in self.vertices:
            if out_minus_in[v] % mod != beta[v] % mod:
                return False, None

        sol = OddOrientationSolution(
            modulus=mod,
            vertices=self.vertices,
            edge_bundles=self.edge_bundles,
            y_by_pair=y_by_pair,
            out_minus_in=out_minus_in,
            beta=beta,
            directions=directions,
        )
        return True, sol

    def is_SZl(self, verbose: bool = False, max_beta: Optional[int] = None) -> Tuple[bool, Optional[Dict[int, int]]]:
        """Decide whether the graph is SZ_l. If not, return a witness infeasible beta.
        max_beta: optional cap for debugging; None means full enumeration.
        """
        cnt = 0
        for beta in self.enumerate_betas():
            cnt += 1
            ok, _ = self.solve_for_beta(beta)
            if verbose and cnt % 200 == 0:
                print(f"Checked betas: {cnt}")
            if not ok:
                return False, beta
            if max_beta is not None and cnt >= max_beta:
                break
        return True, None


def build_graph_from_edges(n: int, edges: List[Tuple[int, int]]) -> nx.MultiGraph:
    Gm = nx.MultiGraph()
    Gm.add_nodes_from(list(range(1, n + 1)))
    for u, v in edges:
        Gm.add_edge(u, v)
    return Gm


def main():
    # ====== Example: edit the graph and l here (l must be odd) ======
    mod = 5
    n = 4
    edges = [(1, 2)]*3+[(1,3)]*3+[(1,4)]*3+[(2,3),(3,4),(4,2),(4,2)]
    # ==========================================================

    Gm = build_graph_from_edges(n, edges)
    solver = SZlOddSolver(Gm, mod)

    print("Graph info:")
    print(f"- vertices: {solver.vertices}")
    print(f"- total edges (with multiplicity): {sum(eb.k for eb in solver.edge_bundles)}")
    print(f"- degrees: {solver.deg}")
    print(f"- l={mod} (odd)")

    is_sz, witness = solver.is_SZl(verbose=True)
    if is_sz:
        print(f"\nConclusion: the graph IS SZ_{mod} ✓")
    else:
        print(f"\nConclusion: the graph is NOT SZ_{mod} ✗")
        print("One infeasible beta (vector):", [witness[v] for v in solver.vertices] if witness else None)

    # ====== Manual beta example (edit below) ======
    # beta must be in 0..l-1 and sum(beta) ≡ 0 mod l
    beta = {v: 0 for v in solver.vertices}
    # ============================================
    ok, sol = solver.solve_for_beta(beta)
    print("\nManual beta:", [beta[v] for v in solver.vertices])
    print("Feasibility:", "feasible" if ok else "infeasible")
    if ok and sol is not None:
        sol.pretty_print()


if __name__ == "__main__":
    main()

