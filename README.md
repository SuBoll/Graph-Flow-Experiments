# Computational Experiments for $SZ_l$ and Related Flow Problems

This repository collects small-scale computational experiments related to
$SZ_l$-orientations, nowhere-zero flows, and closely related connectivity and
flow conditions in graphs.

The code is organized **by topic**, with each subdirectory being
*self-contained* and accompanied by its **own README** explaining the precise
mathematical setting, algorithms, and usage instructions.

---

## Directory structure

### `odd-SZl-4v12e-identifier/`

This folder contains code and documentation for identifying and testing
$SZ_l$-properties **for odd modulus $l$**, focusing on a concrete experimental
case:

- 4-vertex, 12-edge (with multiplicity) undirected multigraphs
- enumeration up to isomorphism
- exhaustive testing of the $SZ_5$ property via a dedicated solver

The folder includes:
- an $SZ_l$ decision and $\beta$-orientation solver for odd $l$
- a graph enumerator with isomorphism reduction
- visualization scripts
- detailed mathematical modeling and implementation notes

Please refer to the README **inside this directory** for full details and
reproducibility instructions.

---

## Future extensions

Additional folders may be added in the future, for example:

- experiments for **even modulus $SZ_l$**
- variants involving **$Z_k$-connectivity**
- computational studies related to **nowhere-zero $k$-flows** or their generalizations

Each such topic will be placed in a **separate directory** with its own
independent README, so that individual experiments can be read, run, and cited
in isolation.

---

## Scope and intent

This repository is intended primarily for:
- experimental verification
- counterexample search
- reproducibility support for related manuscripts

The implementations prioritize **clarity and determinism** over large-scale
performance, and are meant for small-to-moderate instances.
