## Group connectivity: group-valued nowhere-zero flows

This document describes the **group connectivity** solver and the script that searches for smallest graphs distinguishing Z_4 vs Z_2×Z_2. When a specific finite abelian group $`A`$ is given, we consider **A-connectivity** (whether the graph is A-connected).

**Files:**

- `group_connectivity_solver.py`: decides whether a connected multigraph is **A-connected** for a given finite abelian group $`A`$, and finds a **beta-flow** (orientation + flow values) for a given A-boundary $`\beta`$.
- `find_smallest_group_connected.py`: enumerates simple 2-edge-connected graphs in a configurable vertex range and finds smallest graphs that are Z_2×Z_2–connected but not Z_4–connected, or Z_4–connected but not Z_2×Z_2–connected. (Note: This script needs to be further modified before being enabled)

**Dependencies:** Python 3.8+, `networkx`. No matplotlib/numpy for the solver itself.

---

## 0. Recommended layout

Keep in the same directory:

- `group_connectivity_solver.py`
- `find_smallest_group_connected.py` (optional; imports the solver)
- `README.md` (this file)
- `README_CN.md` (Chinese)

---

## 1. Definitions

Let $`A`$ be a finite abelian group with identity $`0_A`$, and $`G`$ a connected undirected multigraph (no self-loops).

### 1.1 A-boundary

An **A-boundary** on $`G`$ is a map

$$
\beta : V(G) \to A
$$

such that the sum over all vertices is zero in $`A`$:

$$
\sum_{v \in V(G)} \beta(v) = 0_A.
$$

### 1.2 Nowhere-zero flow and beta-flow

Fix an orientation of each edge. For an oriented edge $`e = (u \to v)`$, a **flow value** is an element $`f(e) \in A \setminus \{0_A\}`$ (nowhere-zero).

The **net flow** at a vertex $`v`$ is

$$
\mathrm{net}(v) = \sum_{\text{out from } v} f(e) - \sum_{\text{in to } v} f(e) \in A.
$$

A **beta-flow** is an assignment $`f : E(G) \to A \setminus \{0_A\}`$ such that

$$
\mathrm{net}(v) = \beta(v) \quad \text{in } A \quad \text{for every vertex } v.
$$

(Reversing an edge corresponds to replacing $`f(e)`$ by $`-f(e)`$ in $`A`$, so the choice of orientation does not change the existence of a beta-flow.)

### 1.3 A-connected

The graph $`G`$ is **A-connected** if for **every** A-boundary $`\beta`$ there exists a beta-flow.

### 1.4 Group representation in code

By the **classification of finite abelian groups**, every finite abelian group is isomorphic to a direct product of cyclic groups. The code therefore represents $`A`$ in this form:

$$
A \cong \mathbb{Z}_{m_1} \times \mathbb{Z}_{m_2} \times \cdots \times \mathbb{Z}_{m_r},\quad m_i \ge 2.
$$

Elements are tuples $`(x_1,\ldots,x_r)`$ with $`x_i \in \{0,\ldots,m_i-1\}`$. Examples:

- $`[3]`$ → $`A = \mathbb{Z}_3`$, elements $`0,1,2`$.
- $`[2, 2]`$ → $`A = \mathbb{Z}_2 \times \mathbb{Z}_2`$, elements $`(0,0), (0,1), (1,0), (1,1)`$.

---

## 2. Quickstart

### 2.1 Run the solver demo

```bash
python group_connectivity_solver.py
```

This runs the built-in demo (triangle graph, A = Z_3 and A = Z_2×Z_2, and prints a beta-flow for a chosen boundary).

### 2.2 Check a specific beta and print orientation + flow

```python
from group_connectivity_solver import GroupConnectivitySolver, _build_multigraph_from_edges

n = 3
edges = [(1, 2), (2, 3), (3, 1)]
Gm = _build_multigraph_from_edges(n, edges)
solver = GroupConnectivitySolver(Gm, group_moduli=[3])

# A-boundary: sum of beta(v) must be 0 in Z_3. Example: 1+1+1 = 0 (mod 3).
beta = {1: (1,), 2: (1,), 3: (1,)}

feasible, sol = solver.check_beta(beta, verbose=True)
if feasible and sol:
    orientation = sol.get_orientation()   # list of (tail, head)
    flow_values = sol.get_flow_values()  # dict edge_index -> value in A
```

### 2.3 Search for smallest Z_2×Z_2 vs Z_4 distinguishing graphs

Edit `n_min` and `n_max` in `find_smallest_group_connected.py`, then:

```bash
python find_smallest_group_connected.py
```

---

## 3. API (group_connectivity_solver.py)

### 3.1 FiniteAbelianGroup

- **`FiniteAbelianGroup(moduli)`**  
  $`\mathrm{moduli}`$ is a tuple like $`(3)`$ or $`(2, 2)`$. All $`m_i \ge 2`$.

- **`zero()`** — identity in $`A`$.
- **`add(a, b)`**, **`neg(a)`** — group operation and inverse.
- **`elements()`** — list of all elements.
- **`is_zero(a)`** — whether $`a = 0_A`$.
- **`format(a)`** — string for one element (e.g. $`(1,0)`$ for Z_2×Z_2).

### 3.2 FlowSolution

Returned by `solve_for_boundary` when a beta-flow exists.

- **`group`**, **`vertices`**, **`edges`**, **`beta`**, **`flow`** — group, vertex list, oriented edges (tail, head), boundary, and flow dict (edge index → value).
- **`get_orientation()`** — list of $`(\mathrm{tail}, \mathrm{head})`$ for each edge.
- **`get_flow_values()`** — dict edge index → $`f(e) \in A \setminus \{0\}`$.
- **`pretty_print(max_edges=None)`** — print boundary, orientation, and flow values.

### 3.3 GroupConnectivitySolver

**`GroupConnectivitySolver(multigraph, group_moduli)`**

- **`multigraph`**: `networkx.MultiGraph`, connected, no self-loops.
- **`group_moduli`**: e.g. `[3]` or `[2, 2]`.

**Methods:**

- **`enumerate_boundaries()`** — iterates over all A-boundaries (sum = $`0_A`$).
- **`solve_for_boundary(beta)`** — returns a `FlowSolution` if a nowhere-zero beta-flow exists, else `None`.
- **`is_A_connected(verbose=False)`** — returns `(True, None)` if the graph is A-connected, else `(False, counterexample_beta)`.
- **`get_all_infeasible_boundaries(verbose=False)`** — returns a list of all boundaries that have no beta-flow (can be large).
- **`check_beta(beta, verbose=True, max_edges_in_print=None)`** — checks one boundary; if feasible, optionally prints orientation and flow; returns `(True, solution)` or `(False, None)`.

### 3.4 Helper

- **`_build_multigraph_from_edges(n, edges)`** — builds an `nx.MultiGraph` on vertices $`1,\ldots,n`$ from a list of pairs $`(u, v)`$.

---

## 4. find_smallest_group_connected.py

This script:

1. Enumerates **simple** (no multi-edges) **2-edge-connected** graphs on $`n`$ vertices for $`n`$ in $`[\texttt{n\_min}, \texttt{n\_max}]`$.
2. Restricts to graphs with at least $`m \ge \min\{\lceil 4n/3\rceil, \lceil 3(n-1)/2\rceil\}`$ edges.
3. For each graph, tests Z_2×Z_2–connectivity and Z_4–connectivity. If a graph is obtained by adding one edge to another (same vertex set) that is already A-connected, the solver is skipped for that graph.
4. Reports smallest graphs that are Z_2×Z_2–connected but not Z_4–connected, and smallest that are Z_4–connected but not Z_2×Z_2–connected.

Edit `n_min` and `n_max` at the top of `main()` to change the vertex range.

---

## 5. Performance notes

- Group connectivity (A-connectivity for a given group $`A`$) is checked by enumerating all A-boundaries (there are $`|A|^{n-1}`$ many, since the last vertex is determined by the sum condition) and, for each, running a backtracking search for a nowhere-zero flow. Complexity is exponential in $`|V|`$ and $`|E|`$.
- The implementation is intended for **small graphs and small groups** (e.g. experiments). For larger instances, consider theoretical characterizations or more specialized algorithms.


