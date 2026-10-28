"""Input matrix (Methods, Eq. 1): log counts of every gene, weighted by inverse document frequency (IDF), rows of unit length.
The in-place version is the one used in the paper; it changes the CSR matrix it receives and makes no copy."""
import numpy as np
import scipy.sparse as sp


def to_csr32(X):
    """Raw counts as a float32 CSR matrix (copy only when the type or format differs)."""
    X = X if sp.isspmatrix_csr(X) else sp.csr_matrix(X)
    return X if X.dtype == np.float32 else X.astype(np.float32)


def idf_matrix_inplace(X, library_size=False, idf=True):
    """X: float32 CSR of raw counts (cells x genes), modified in place.
    library_size: scale every cell to 10,000 counts first (used with idf=False for the 5,000-HVG mode).
    Weight of gene g: log((m+1)/(d_g+1)) / max over genes, d_g = number of cells that detect g; then log1p, weight, l2 rows."""
    assert sp.isspmatrix_csr(X) and X.dtype == np.float32, "pass a float32 CSR matrix (see to_csr32)"
    n, p = X.shape
    if library_size:
        rs = np.add.reduceat(X.data, X.indptr[:-1]) if X.nnz else np.zeros(n, np.float32)
        rs[np.diff(X.indptr) == 0] = 0
        f = (1e4 / np.maximum(rs, 1)).astype(np.float32)
        X.data *= np.repeat(f, np.diff(X.indptr))
    ndet = np.bincount(X.indices[X.data != 0], minlength=p).astype(np.float64)
    w = np.log((n + 1) / (ndet + 1)); w = (w / w.max()).astype(np.float32)
    if not idf:
        w = np.ones_like(w)
    np.log1p(X.data, out=X.data); X.data *= w[X.indices]
    sq = np.zeros(n, np.float64); nz = np.diff(X.indptr) > 0
    sq[nz] = np.add.reduceat(X.data.astype(np.float64) ** 2, X.indptr[:-1][nz])
    nrm = np.sqrt(sq); nrm[nrm == 0] = 1
    X.data /= np.repeat(nrm, np.diff(X.indptr)).astype(np.float32)
    return X


def hvg_index(X, n_top=5000):
    """Indices of the n_top highly variable genes (scanpy, normalized to 1e4 and log1p, default flavor)."""
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(X.copy()); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n_top)
    return np.where(a.var.highly_variable.values)[0]
