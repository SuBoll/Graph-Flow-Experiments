## SZ_5 (odd modulus) experiments: solver + enumerator

This folder accompanies computational experiments and reproducibility notes for a manuscript/report. It contains two scripts (intended to live in the same directory):

- `szl_odd_solver.py`: an $SZ_l$ decision and $\beta$-orientation solver for **odd** modulus $l$ (this project uses $l=5$).
- `generate_nonisomorphic_4v12e.py`: enumerates constrained 4-vertex, 12-edge (with multiplicity) undirected multigraphs, removes isomorphic duplicates, renders an overview figure, and tests each graph for $SZ_5$ by calling `szl_odd_solver.py`.

This README is the **single main document**: it first explains how the two scripts fit together, then gives a detailed account of the mathematical modeling and the implementation workflow of each script (including DFS pruning and isomorphism reduction), so results can be reviewed and reproduced.

> Note: you may see `libpng warning: iCCP: known incorrect sRGB profile` at runtime. This is typically a benign PNG color-profile warning and does not affect saving or displaying the figure.

---

## 0. Recommended layout

Keep these files in the same directory:

- `szl_odd_solver.py`
- `generate_nonisomorphic_4v12e.py`
- `README.md` (Chinese)
- `README_EN.md` (this file)

`generate_nonisomorphic_4v12e.py` dynamically imports `szl_odd_solver.py` from the same directory to access `SZlOddSolver`, so they must be co-located unless you adjust the import path.

---

## 1. Requirements

- Python 3.8+
- `networkx`
- `matplotlib`
- `numpy`

Install example:

```bash
python -m pip install networkx matplotlib numpy
```

---

## 2. Quickstart (recommended)

### 2.1 Run enumeration + overview figure + SZ_5 testing

```bash
python generate_nonisomorphic_4v12e.py
```

This will:

- open a Matplotlib window showing the overview grid (all non-isomorphic representatives)
- save an overview PNG to the current working directory:
  - `nonisomorphic_4v12e_SZ_5_overview.png`
- print $SZ_5$ testing results to the console; if a graph is not $SZ_5$, it prints a witness boundary $\beta$ (as a vector)

### 2.2 Run the solver demo only

```bash
python szl_odd_solver.py
```

The script contains a built-in `main()` demo. You can edit `mod/n/edges/beta` to test different graphs and boundaries.

---

## 3. `szl_odd_solver.py`: problem statement

### 3.1 Valid boundaries $\beta$

Let $G$ be a connected undirected multigraph (parallel edges allowed, no self-loops) and let $l$ be a positive odd integer.

A function $\beta:V(G)\to\mathbb{Z}_l$ is a valid boundary if:

- $\beta(v)\in\{0,1,\dots,l-1\}$ for all $v$
- $\sum_{v\in V}\beta(v)\equiv 0\pmod l$

The sum condition is necessary since $\sum_v(\text{out}-\text{in})=0$ for any orientation.

### 3.2 $\beta$-orientation and $SZ_l$

A $\beta$-orientation is an orientation of all edges such that for each vertex $v$:

$$
(\operatorname{outdeg}(v)-\operatorname{indeg}(v))\equiv \beta(v)\pmod l
$$

The graph $G$ is $SZ_l$ if such an orientation exists for **every** valid boundary $\beta$.

---

## 4. `szl_odd_solver.py`: modeling (as implemented)

### 4.1 Bundling parallel edges: one integer $y$ per vertex pair

For an unordered pair $\{u,v\}$ (normalized as $u<v$) with multiplicity $k$, define an integer variable $y\in\{0,1,\dots,k\}$ as the number of edges oriented $u\to v$. Then:

- net contribution at $u$ (out-in) is $2y-k$
- net contribution at $v$ is $-(2y-k)$

Thus we search one integer variable per vertex pair (domain size $k+1$) rather than enumerating $2^k$ edge directions.

### 4.2 Signs and constant terms $C_v$

For each bundle $e=(u,v)$ (normalized $u<v$), define:

- $\operatorname{sign}(u,e)=+1$
- $\operatorname{sign}(v,e)=-1$

For each vertex:

$$
S(v)=\sum_{e\ni v}\operatorname{sign}(v,e)\cdot y_e
$$

Expanding $(2y_e-k_e)$ yields the congruence:

$$
C_v+2S(v)\equiv \beta(v)\pmod l,\qquad
C_v=\sum_{e\ni v}\operatorname{sign}(v,e)\cdot(-k_e)
$$

### 4.3 Odd modulus trick: divide by 2 in $\mathbb{Z}_l$

When $l$ is odd, 2 is invertible in $\mathbb{Z}_l$. Let $2^{-1}$ be its inverse. Then:

$$
S(v)\equiv 2^{-1}(\beta(v)-C_v)\pmod l
$$

In the code, the right-hand side is precomputed as `target_residue[v_idx]` (aligned with the sorted vertex list `vertices`).

---

## 5. `szl_odd_solver.py`: implementation workflow (by functions)

### 5.1 Core data structures

- `EdgeBundle(u, v, k)`: a normalized vertex pair with multiplicity $k$ (always `u < v`).
- `OddOrientationSolution`: stores a solution, including:
  - `y_by_pair[(u,v)]`: the $y$ value per bundle
  - `out_minus_in[v]`: integer out-in values (not reduced mod $l$)
  - `directions`: expanded per-edge directions `(tail, head)` (useful for verification/replay)

### 5.2 `SZlOddSolver.__init__`: preprocessing

Initialization performs:

- input validation (odd $l$, no self-loops, connected)
- `vertices = sorted(nodes)` and `index_of_vertex`
- bundle collection `_collect_edge_bundles()`
- sign structure `_build_signs()` (incident bundles and $\pm1$ per vertex)
- degree computation (counting multiplicity)
- constant term vector `C_vec`
- inverse `inv2 = 2^{-1} (mod l)`

### 5.3 `enumerate_betas()`: enumerate all valid $\beta$

For $n$ vertices, it enumerates the first $n-1$ values (there are $l^{n-1}$ choices) and determines the last value uniquely from $\sum\beta\equiv 0\pmod l$. This avoids a full $l^n$ enumeration.

### 5.4 `solve_for_beta(beta)`: DFS + pruning (core)

High-level steps:

1. validate `beta` (vertex set, value range, sum condition)
2. compute target residues `target_residue[v]`
3. define domains $y_e\in\{0,\dots,k_e\}$ for each bundle
4. order variables by increasing domain size (small $k_e$ first)
5. DFS assignment while maintaining `partial_sum[v]` (integer partial sums of $S(v)$)
6. prune using “interval reachability under congruence”:
   - compute the remaining possible contribution interval $[L,U]$ for each vertex
   - let `need ≡ target_residue[v] - partial_sum[v] (mod l)`
   - prune if no $t\in[L,U]$ satisfies $t\equiv \text{need}\pmod l$
7. on success, assemble the expanded edge directions and verify `(out-in) mod l == beta`

### 5.5 `is_SZl(...)`: decision and witness

Enumerate all valid $\beta$ and call `solve_for_beta`:

- if any $\beta$ is infeasible, return `(False, beta)` as a witness
- otherwise return `(True, None)`

---

## 6. `generate_nonisomorphic_4v12e.py`: what it does

This script packages “dataset generation + visualization + batch testing”:

1. enumerate 4-vertex, 12-edge (with multiplicity) multigraphs under constraints
2. remove isomorphic duplicates (keep representatives)
3. draw and save an overview grid figure (PNG)
4. test each representative for $SZ_5$ using `SZlOddSolver(..., 5)` and print witnesses when found

---

## 7. `generate_nonisomorphic_4v12e.py`: implementation workflow

### 7.1 Exact enumeration space

Vertices are fixed as $\{1,2,3,4\}$. We consider the 6 unordered pairs:

$$
(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
$$

For each pair, assign a multiplicity $k_{uv}\in\{0,1,2,3\}$ and filter by:

- total multiplicity: $\sum k_{uv}=12$
- minimum degree (counting multiplicity): each degree is $\ge 4$
- connectivity of the underlying simple graph (treat $k_{uv}>0$ as an edge)

### 7.2 Isomorphism reduction: canonical key over all $4!$ permutations

The script implements a deterministic canonicalization:

- enumerate all $4!$ vertex permutations
- compute the induced 6-tuple under each permutation
- take the lexicographically smallest tuple as the canonical key
- deduplicate by this key

### 7.3 Plotting: consistent visual scale across subplots

With curved arc patches for parallel edges, Matplotlib can autoscale subplots differently, making some graphs look larger/smaller. The script fixes this by:

- using a fixed radial layout radius
- fixing `xlim/ylim` to the same range for every subplot

### 7.4 Output: save overview PNG

The script saves the overview figure to the current working directory:

- `nonisomorphic_4v12e_SZ_5_overview.png`

and prints the saved path.

### 7.5 Batch $SZ_5$ testing and console output

For each representative:

- build an `nx.MultiGraph`
- call `SZlOddSolver(Gm, 5).is_SZl(verbose=False)`
- if infeasible, print:
  - graph index
  - edge multiplicities (6 pairs)
  - vertex degrees
  - a witness $\beta$ printed as a vector in `solver.vertices` order:
    - `[witness[v] for v in solver.vertices]`

---

## 8. Performance and reproducibility notes

- **Solver complexity**: the number of valid boundaries is $l^{n-1}$; each DFS is exponential in the worst case. This implementation targets small-to-moderate instances for experimental verification.
- **Determinism**:
  - `szl_odd_solver.py`: vertices and bundles are sorted; given the same input, search order and output are deterministic (when a solution exists)
  - `generate_nonisomorphic_4v12e.py`: the canonical-key approach is fully deterministic

