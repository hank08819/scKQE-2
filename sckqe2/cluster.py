"""15-nearest-neighbor graph with UMAP fuzzy weights and Leiden at a given number of clusters (Methods, Scoring)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def knn(F, k=15, chunk=512, device=None):
    """Exact euclidean nearest neighbors; each cell is its own first neighbor (scanpy convention).
    GPU (torch) when available, otherwise scikit-learn brute force."""
    F = np.ascontiguousarray(F, np.float32); n = len(F)
    try:
        import torch
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dev.type != "cuda":
            raise ImportError
        X = torch.as_tensor(F, device=dev); sq = (X * X).sum(1)
        I = np.empty((n, k), np.int64); D = np.empty((n, k), np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n); rows = torch.arange(s, e, device=dev)
            d2 = sq[s:e, None] - 2 * X[s:e] @ X.T + sq[None, :]
            d2[torch.arange(e - s, device=dev), rows] = float("inf")
            v, i = torch.topk(d2, k - 1, dim=1, largest=False)
            I[s:e, 0] = rows.cpu().numpy(); D[s:e, 0] = 0.0
            I[s:e, 1:] = i.cpu().numpy(); D[s:e, 1:] = v.clamp(min=0).sqrt().cpu().numpy()
        return I, D
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        D, I = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(F).kneighbors(F)
        return I.astype(np.int64), D.astype(np.float32)


def knn_graph(F, k=15, **kw):
    from umap.umap_ import fuzzy_simplicial_set
    I, D = knn(F, k, **kw); n = len(I)
    G, _, _ = fuzzy_simplicial_set(sp.coo_matrix((n, 1)), k, None, "euclidean", knn_indices=I, knn_dists=D,
                                   set_op_mix_ratio=1.0, local_connectivity=1.0)
    return G.tocsr()


def leiden(G, resolution=1.0, seed=0):
    import anndata as ad
    import scanpy as sc
    A = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(G.shape[0])]))
    sc.tl.leiden(A, resolution=resolution, key_added="L", flavor="igraph", n_iterations=2, directed=False, adjacency=G,
                 random_state=seed)
    return A.obs["L"].astype(int).values.copy()


def leiden_at_k(G, k, seed=0, steps=30, lo=0.02, hi=3.0):
    """Bisect the resolution until Leiden returns k clusters (or the closest number found in `steps` steps)."""
    best = None
    for _ in range(steps):
        res = (lo + hi) / 2; lab = leiden(G, res, seed); kk = len(np.unique(lab))
        if best is None or abs(kk - k) < abs(best[0] - k):
            best = (kk, lab, res)
        if kk == k:
            break
        lo, hi = (res, hi) if kk < k else (lo, res)
    return best[1], best[2]
