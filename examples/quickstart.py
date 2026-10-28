"""Quickstart: cluster an atlas with scKQE-2 and check that a second run is bit-identical."""
import sys
import numpy as np
import anndata as ad
import sckqe2

A = ad.read_h5ad(sys.argv[1])                       # raw counts in .raw.X or .X; annotated types in obs['cell_type']
y = A.obs["cell_type"].astype(str).values
k = len(np.unique(y))
gene_set, delta, h, L = sckqe2.choose_gene_set(A.raw.X if A.raw is not None else A.X, k)
print(f"gene-set rule: Delta = {delta:.2f} -> {gene_set}")
m1 = sckqe2.kqe2(A, n_clusters=k, gene_set=gene_set)
first = A.obs["kqe2"].values.copy()
sckqe2.kqe2(A, n_clusters=k, gene_set=gene_set)
print("second run identical:", np.array_equal(first, A.obs["kqe2"].values))
