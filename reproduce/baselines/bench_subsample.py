"""Write the fixed 150,000-cell subsample of the paper (RandomState(0) over all cells, drop nan/unknown labels and types
with <10 cells) as raw counts: <out>/<tag>/counts.mtx (cells x genes), genes.tsv, cells.tsv, labels.tsv, and sub.h5ad.
Usage: python bench_subsample.py <h5ad> <label_col> <tag> <outdir>   env KQE_SUB (150000)"""
import os, sys, numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio, anndata as ad
path, labcol, tag, out = sys.argv[1:5]; o = os.path.join(out, tag); os.makedirs(o, exist_ok=True)
a = ad.read_h5ad(path, backed="r"); n0 = a.n_obs; sub = int(os.environ.get("KQE_SUB", 150000))
sel = np.sort(np.random.RandomState(0).choice(n0, sub, replace=False)) if 0 < sub < n0 else np.arange(n0)
b = a[sel].to_memory(); y = b.obs[labcol].astype(str).values
X = sp.csr_matrix(b.raw.X if b.raw is not None else b.X); var = b.raw.var if b.raw is not None else b.var
sym = np.asarray(var["feature_name"] if "feature_name" in var.columns else var.index).astype(str)
keep = (y != "nan") & (y != "unknown"); cls, cnt = np.unique(y[keep], return_counts=True); ok = keep & np.isin(y, cls[cnt >= 10])
X, y, cells = X[ok], y[ok], np.asarray(b.obs_names)[ok]; del a, b
X.data = np.round(X.data)                                    # integer counts for scVI / Seurat
batch = None
A = ad.AnnData(X.astype(np.float32), obs=pd.DataFrame({"label": y}, index=cells), var=pd.DataFrame({"gene_name": sym}, index=np.asarray(var.index).astype(str)))
A.var_names_make_unique(); A.write_h5ad(os.path.join(o, "sub.h5ad"))
sio.mmwrite(os.path.join(o, "counts.mtx"), X.T.tocoo())     # genes x cells for R
pd.Series(A.var_names).to_csv(os.path.join(o, "genes.tsv"), index=False, header=False)
pd.Series(cells).to_csv(os.path.join(o, "cells.tsv"), index=False, header=False)
pd.Series(y).to_csv(os.path.join(o, "labels.tsv"), index=False, header=False)
print(f"{tag}: {X.shape[0]} cells x {X.shape[1]} genes, {len(np.unique(y))} types -> {o}", flush=True)
