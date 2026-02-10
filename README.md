# GraphFlowExperiments

This repository collects small-scale computational experiments related to graph flows, orientations, and related problems in graphs.

The code is organized **by topic**, with each subdirectory being
*self-contained* and accompanied by its **own README** explaining
the precise mathematical setting, algorithms, and usage instructions.

---

## Directory structure

### `odd-SZl-4v-identifier/`

This folder contains code and documentation for identifying and testing
$SZ_l$-properties **for odd modulus $l$**, focusing on 4-vertex multigraphs:

- 4-vertex, $3(l-1)$-edge (with multiplicity) undirected multigraphs for a given odd $l$
- enumeration up to isomorphism (min degree $\ge l-1$, max multiplicity per pair $\le l-2$, connected)
- exhaustive testing of the $SZ_l$ property via a dedicated solver

See the README **inside this directory** for full details and reproducibility instructions.

### `general_szl_identifier/`  

This folder contains scripts for **general modulus $l$** (odd or even), complementary to the odd-modulus version:

- `$szl_solver.py$`: $SZ_l$ decision and $\beta$-orientation solver for arbitrary positive $l$  
- `$generate_nonisomorphic_szl.py$`: enumerates $n$-vertex, $m$-edge multigraphs under degree and multiplicity constraints, removes isomorphic duplicates, tests each graph for $SZ_l$, and generates overview figures  

See the README **inside this directory** for step-by-step modeling, implementation, and usage instructions.

---

## Future extensions

Additional folders may be added in the future, for example:

- computational studies related to **$Z_k$-connectivity**
- **nowhere-zero $k$-flows** or other graph flow generalizations

Each topic will reside in a **separate directory** with its own README so
that experiments can be read, run, and cited independently.

---

## Scope and intent

This repository is intended primarily for:

- experimental verification
- counterexample search
- reproducibility support for related manuscripts

The implementations prioritize **clarity and determinism** over large-scale
performance, targeting small-to-moderate instances suitable for
mathematical experimentation and verification.
