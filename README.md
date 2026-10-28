# scKQE-2

Reproducible quantum clustering of atlas-scale single-cell data on the whole transcriptome.

scKQE-2 turns every cell into an exact quantum state built from its whole transcriptome and clusters the cells in
quantum-state space. The encoding has no entangling gate, so every quantum feature has a closed form, no step is trained and
no step draws a random number: the same counts give bit-for-bit identical clusters on every run on the same machine.

This package implements the method of the paper *Reproducible quantum clustering of atlas-scale single-cell data*
(H. Han) exactly as used for its results.

## Install

```bash
pip install scKQE2                 # CPU
pip install "scKQE2[gpu]"          # + PyTorch: GPU encoding, nearest neighbours and exact quantum-state search
pip install "scKQE2[hardware]"     # + Qiskit: run the encoding on a quantum processor
pip install "scKQE2[test]"; pytest # 12 tests, about 30 s
```

Python 3.9 or later. Dependencies: numpy, scipy, scikit-learn, pandas, anndata, scanpy, umap-learn, igraph, leidenalg.

## Quick start

```python
import anndata as ad, sckqe2
adata = ad.read_h5ad("atlas.h5ad")                # raw counts in adata.raw.X or adata.X
k = adata.obs["cell_type"].nunique()              # labels are used only to set k
sckqe2.kqe2(adata, n_clusters=k)                  # adata.obs["kqe2"], adata.obsm["X_kqe2"], adata.uns["kqe2"]
```

or, on a count matrix,

```python
model = sckqe2.KQE2()
labels = model.fit_predict(counts, n_clusters=k)  # model.gene_set_used_, model.delta_, model.scores_, model.features_
```

Command line: `sckqe2 atlas.h5ad --label cell_type --out clusters.csv` (`--donor donor_id` for quantum alignment).

## What the default pipeline does (the configuration of Table 1)

| Step | Function | Paper |
|---|---|---|
| Gene-set rule: Delta = h_PA - log2 k; whole transcriptome with IDF if Delta <= 1.15, else 5,000 HVGs | `choose_gene_set` | Methods |
| IDF-weighted log counts, rows of unit length | `idf_matrix_inplace` | Eq. 1, Methods |
| Exact decomposition: ARPACK from a fixed start vector, 132 whitened scores, no random number | `exact_scores` | Methods |
| Entanglement-free encoding: 22 blocks of 6 qubits, 3 layers of R_Y(2^-t s x) R_Z(s x), s = 0.19; 726 exact features | `encode`, `bloch_vectors` | Fig. 2a, Theorems 2-4 |
| 15-nearest-neighbour graph with UMAP fuzzy weights | `knn_graph` | Methods |
| Precision Shrinkage Clustering on whole-transcriptome atlases; Leiden at k on 5,000-HVG atlases (Leiden seed 0) | `leiden_psc`, `leiden_at_k` | Methods |
| Optional deterministic donor correction (quantum alignment) | `quantum_align` | Fig. 2d, Methods |

`KQE2(gene_set="all" | "hvg5000", psc=True | False)` overrides the rules.

## Exact quantum-state geometry (Fig. 3)

```python
I, D, certified = model.certified_state_neighbors(k=15)   # exact nearest quantum states of every cell
xy = sckqe2.state_map(I, D)                                # map built from exact state fidelity alone
sckqe2.neighbor_purity(I, labels)                          # share of the nearest states in the own cluster
```

The fidelity of two cells is F = prod_q (1 + r_q . r'_q)/2 over their Bloch vectors. Candidates are taken by feature distance,
and the lower bound of Theorem 4, -log F >= ||f - f'||^2 / 4, proves that no other cell can be closer; cells without this
certificate are searched exhaustively, so every neighbour is exact. All 888,263 nuclei of the Siletti atlas take 100 s on one GPU.

## Quantum hardware

```python
from sckqe2.hardware import block_circuit, observables, exact_expectations
qc = block_circuit(scores[0, :6])         # one block of one cell; run with qiskit-ibm-runtime EstimatorV2
exact = exact_expectations(scores[0, :6]) # the exact values every hardware measurement is checked against
```

## Reproducing the paper

`reproduce/` downloads the atlases from CZ CELLxGENE Discover (dataset version ids of Extended Data Table 8) and reruns the
KQE-2 results, the quantum-state maps and the rare-type marker check; `reproduce/baselines/` holds the scripts used for
scanpy, Seurat, scVI and scGPT. See `reproduce/README.md`.

## Tests

`tests/test_sckqe2.py` checks the closed form against a brute-force statevector simulation, the two encoding backends against each
other, the exact decomposition against a dense SVD, bit-identical reruns, the rotation identity behind quantum alignment, the
data entropy, Precision Shrinkage Clustering, the certified nearest states against an exhaustive search, the lower bound of
Theorem 4, and the Qiskit circuit against the closed form.

## Citation

Han, H. Reproducible quantum clustering of atlas-scale single-cell data (2026). See `CITATION.cff`.

Author: Henry Han, Baylor University (Henry_Han@Baylor.edu). License: MIT.
