## Flow index: nowhere-zero flows in the cycle space

This document describes the **flow index** script `compute_flow_index.py`, which computes the minimum maximum edge norm of a nowhere-zero flow in the cycle space of a connected multigraph.

**File:**

- `compute_flow_index.py`: assigns cycle-space coefficients to each edge, runs constrained optimization to find a nowhere-zero flow with minimal maximum norm, and outputs the flow value $`r = M + 1`$.

**Dependencies:** Python 3.8+, `networkx`, `numpy`, `scipy`, `matplotlib`.

---

## 0. Recommended layout

Keep in the same directory:

- `compute_flow_index.py`
- `README.md` (this file)
- `README_CN.md` (Chinese)

---

## 1. Mathematical background

Let $`G`$ be a connected undirected multigraph with $`n`$ vertices and $`m`$ edges. The **cycle space** has dimension $`k = m - n + 1`$. Each edge $`e`$ can be represented by a coefficient vector $`c_e \in \mathbb{Z}^k`$ in a chosen basis of the cycle space.

We seek a map $`X : \mathbb{R}^k \to \mathbb{R}^d`$ such that for each edge $`e`$,

$$
f_e = \sum_{i=1}^{k} c_{e,i} \, X_i \in \mathbb{R}^d
$$

satisfies:

- **Nowhere-zero**: $`\|f_e\|_p \ge 1`$ for all edges $`e`$,
- **Flow conservation**: By construction from the cycle space, the net flow at each vertex is zero (Out $`-`$ In $`= 0`$).

The **objective** is to minimize $`M = \max_e \|f_e\|_p`$ over all feasible $`X`$. The **flow value** is $`r = M + 1`$.

---

## 2. Quickstart

### 2.1 Run the demo

```bash
python compute_flow_index.py
```

This runs the built-in example (Petersen graph, $`d=2`$, $`p=2`$, L2 norm), prints the graph structure, visualizes the graph, runs the optimization, and outputs the edge vectors and flow conservation check.

### 2.2 Use in your own code

```python
import networkx as nx
from compute_flow_index import (
    assign_edge_vectors_multigraph,
    solve_with_constraints,
    visualize_multigraph,
)

# Build a MultiGraph with edges (u, v, key=idx, idx=idx)
MG = nx.MultiGraph()
MG.add_nodes_from(range(10))
edges = [(i, (i+1)%5) for i in range(5)] + [(i,i+5) for i in range(5)] + [(i+5,(i+2)%5+5) for i in range(5)]
for idx, (u, v) in enumerate(edges):
    MG.add_edge(u, v, key=idx, idx=idx)

# Assign cycle-space coefficients and solve
edge_vectors, tree_edge_keys = assign_edge_vectors_multigraph(MG)
best_M, final_edge_vectors = solve_with_constraints(MG, edge_vectors, d=2, p=2, repeats=20, maxiter=1000)

print(f"Flow value r = {best_M + 1:.6f}")
```

---

## 3. Parameters

In `main_example()`, you can adjust:

- **`d`**: Dimension of the flow vectors (default: 2).
- **`p`**: Norm order for $`\|\cdot\|_p`$ (default: 2, i.e. Euclidean).
- **`repeats`**: Number of random restarts for the optimizer (default: 40).
- **`maxiter`**: Maximum iterations per SLSQP run (default: 5000).

You can also change the graph by uncommenting one of the example graph constructions (wheel, cycle, dipole) or defining your own `edges` list.

---

## 4. API

### 4.1 assign_edge_vectors_multigraph

**`assign_edge_vectors_multigraph(MG: nx.MultiGraph) -> (edge_vectors, tree_edge_keys)`**

- **`MG`**: Connected `networkx.MultiGraph` without self-loops. Each edge should have `key` and optionally `idx` in its data.
- **Returns**: `edge_vectors` is a dict mapping `(u, v, key)` to an integer vector of length $`k = m - n + 1`$ (cycle-space coefficients). `tree_edge_keys` lists the edges that belong to the spanning tree.

### 4.2 solve_with_constraints

**`solve_with_constraints(MG, edge_vectors, d=2, p=2, repeats=20, maxiter=1000)`**

- **`MG`**, **`edge_vectors`**: From `assign_edge_vectors_multigraph`.
- **`d`**: Dimension of flow vectors.
- **`p`**: Norm order (e.g. 2 for L2).
- **`repeats`**: Random restarts.
- **`maxiter`**: SLSQP iterations per run.
- **Returns**: `(best_M, final_edge_vectors)` where `best_M` is the minimum maximum edge norm and `final_edge_vectors` maps each edge to its $`\mathbb{R}^d`$ flow vector.

### 4.3 visualize_multigraph

**`visualize_multigraph(G, tree_edges, edge_vectors)`**

Draws the graph structure (Petersen uses a custom layout; others use Kamada–Kawai).

### 4.4 Utilities

- **`print_graph_structure(G)`** — prints vertices, edges, and cycle-space dimension.
- **`print_edge_vectors(MG, final_edge_vectors)`** — prints each edge’s flow vector and norm.
- **`print_flow_conservation(MG, final_edge_vectors, p=2)`** — verifies flow conservation at each vertex.

---

## 5. Example graphs

The script includes commented examples:

- **Petersen graph** (default): 10 vertices, 15 edges, $`k=6`$.
- **Wheel graph** $`W_k`$: hub + cycle.
- **Cycle** $`C_k`$: single cycle.
- **Dipole** $`Mu_k`$: two vertices with $`k`$ parallel edges.

---

## 6. Performance notes

- The optimization uses SLSQP with inequality constraints $`\|f_e\|_p \ge 1`$ and minimizes $`\max_e \|f_e\|_p`$. Multiple random restarts improve the chance of finding a good solution.
- Complexity is dominated by the optimizer; the cycle-space assignment and visualization are $`O(m)`$ and $`O(m^2)`$ respectively.
- For larger graphs, increase `repeats` and `maxiter` as needed.

