"""Locked gene-set rule (fixed on 2026-10-16 before testing on new atlases).

Delta = h_PA - log2(k), where h_PA is the data entropy of the singular values of the directions above a gene-permutation
null (parallel analysis) on 10,000 cells, and k is the requested number of clusters. Delta > 1.15 -> 5,000 HVGs without IDF;
otherwise the whole transcriptome with IDF. The data entropy of singular values s_1..s_r is H = -sum u_i log2 u_i, u_i = s_i / sum s_j."""
import numpy as np
from .preprocess import to_csr32, idf_matrix_inplace

THRESHOLD = 1.15


def data_entropy(s):
    u = np.asarray(s, np.float64); u = u[u > 0] / u[u > 0].sum()
    return float(-(u * np.log2(u)).sum())


def _centered_sv(M):
    m = np.asarray(M.mean(0)).ravel(); G = (M @ M.T).toarray(); r = np.asarray(M @ m).ravel()
    G = G - r[:, None] - r[None, :] + float(m @ m)
    return np.sqrt(np.clip(np.linalg.eigvalsh(G)[::-1], 0, None))


def signal_entropy(X, n_cells=10000, n_perm=3, seed=0):
    """h_PA of the IDF-weighted matrix of up to n_cells cells; the null permutes each gene independently across cells."""
    rs = np.random.RandomState(seed); X = to_csr32(X)
    sel = np.sort(rs.choice(X.shape[0], min(n_cells, X.shape[0]), replace=False))
    Xi = idf_matrix_inplace(X[sel].copy()).astype(np.float64); s = _centered_sv(Xi)
    nulls = []
    for _ in range(n_perm):
        C = Xi.tocsc(copy=True)
        for g in range(C.shape[1]):
            a, b = C.indptr[g], C.indptr[g + 1]
            if b > a:
                C.indices[a:b] = rs.choice(C.shape[0], b - a, replace=False)
        C.sort_indices(); nulls.append(_centered_sv(C.tocsr()))
    null = np.mean(nulls, 0); L = int(np.argmax(~(s > null))) if np.any(~(s > null)) else len(s)
    return data_entropy(s[:L]), L


def choose_gene_set(X, k, n_cells=10000, threshold=THRESHOLD):
    """Returns ('all' or 'hvg5000', Delta, h_PA, L_PA)."""
    h, L = signal_entropy(X, n_cells)
    delta = h - np.log2(k)
    return ("hvg5000" if delta > threshold else "all"), float(delta), h, L
