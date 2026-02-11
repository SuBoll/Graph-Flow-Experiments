# Graph-Flow-Experiments

This repository collects small-scale computational experiments related to
graph flows, orientations, and group-valued connectivity problems in graphs,
including $SZ_l$-orientations, group-valued nowhere-zero flows, and
continuous flow index optimization in the cycle space.

The code is organized **by topic**, with each subdirectory being
*self-contained* and accompanied by its **own README** explaining
the precise mathematical setting, algorithms, and usage instructions.

---

## Directory structure

### `flow-index/`

This folder contains code for computing the **flow index** of a connected
multigraph via continuous optimization in the cycle space.

Given a connected graph $G$ with cycle-space dimension
$k = m - n + 1$, the script:

- assigns integer cycle-space coefficients to each edge
- constructs edge flow vectors in $\mathbb{R}^d$
- enforces the **nowhere-zero constraint** $\|f_e\|_p \ge 1$
- minimizes the maximum edge norm
- outputs the flow value $r = M + 1$

The folder includes:

- `compute_flow_index.py`: main solver and visualization tools
- detailed documentation of the mathematical formulation
- reproducible examples (Petersen graph, wheel, cycle, dipole)

See the README **inside this directory** for full mathematical background,
API description, and usage instructions.

---

### `odd-SZl-4v-identifier/`

This folder contains code and documentation for identifying and testing
$SZ_l$-properties **for odd modulus $l$**, focusing on 4-vertex multigraphs:

- 4-vertex, $3(l-1)$-edge (with multiplicity) undirected multigraphs for a given odd $l$
- enumeration up to isomorphism (min degree $\ge l-1$, max multiplicity per pair $\le l-2$, connected)
- exhaustive testing of the $SZ_l$ property via a dedicated solver

See the README **inside this directory** for full details and reproducibility instructions.

---

### `general_szl_identifier/`

This folder contains scripts for **general modulus $l$** (odd or even),
complementary to the odd-modulus version:

- `szl_solver.py`: an $SZ_l$ decision and $\beta$-orientation solver for arbitrary positive $l$
- `generate_nonisomorphic_szl.py`: enumerates $n$-vertex, $m$-edge multigraphs under degree and multiplicity constraints, removes isomorphic duplicates, tests each graph for $SZ_l$, and generates overview figures

See the README **inside this directory** for step-by-step modeling,
implementation details, and usage instructions.

---

### `Z_l-connected-identifier/`

This folder contains code for **group connectivity** and **group-valued
nowhere-zero flows**.

Given a finite abelian group $A$ (e.g. $\mathbb{Z}_l$ or
$\mathbb{Z}_2 \times \mathbb{Z}_2$), the scripts decide whether a graph is
**A-connected**, i.e., whether every valid $A$-boundary admits a
nowhere-zero $A$-flow.

The folder includes:

- a solver for deciding A-connectivity and constructing beta-flows
- scripts for searching smallest graphs that distinguish different groups
  (e.g. $\mathbb{Z}_4$ vs $\mathbb{Z}_2 \times \mathbb{Z}_2$)
- detailed definitions, modeling choices, and implementation notes

See the README **inside this directory** for precise definitions,
algorithmic details, and usage examples.

---

## Future extensions

Additional folders may be added in the future, for example:

- further studies of **continuous flow invariants**
- computational experiments on **nowhere-zero flows**
- extended investigations of **group connectivity**
- related problems in **graph orientations**, **cycle space structure**, and
  flow constraints

Each topic will be placed in a **separate directory** with its own
independent README, so that experiments can be read, run, and cited
in isolation.

---

## Scope and intent

This repository is intended primarily for:

- experimental verification
- counterexample search
- reproducibility support for related manuscripts
- exploration of structural flow invariants

The implementations prioritize **clarity and determinism** over large-scale
performance, and are intended for small-to-moderate instances suitable for
mathematical experimentation and verification.
