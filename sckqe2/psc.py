"""Precision Shrinkage Clustering (PSC; Methods).

On the 15-nearest-neighbor graph of the quantum features, Leiden is bisected to k' = ceil(1.5 k) clusters (seed 0). With Pi the
membership matrix and A the weighted adjacency matrix, C = Pi^T A Pi holds the edge weight between every pair of clusters.
The pair (a, b) with the largest normalized connectivity C_ab / sqrt(v_a v_b), v_a = sum_b' C_ab', is merged, C and v are
updated, and the step is repeated until k clusters remain. Every operation is deterministic and acts on a k' x k' matrix,
so the step adds seconds to a run. Pieces of an abundant type are joined again, while weakly connected small types keep
their own clusters.

KQE-2 applies PSC whenever the gene-set rule selects the whole transcriptome; on atlases for which it selects 5,000 HVGs,
Leiden at k is used directly (see sckqe2.rule.choose_gene_set and KQE2(psc="auto")).
"""
import numpy as np
import scipy.sparse as sp
from .cluster import leiden_at_k

FACTOR = 1.5


def merge_to_k(labels, A, k):
    """Greedily merge the clusters of `labels` on the graph A until k clusters remain (normalized connectivity)."""
    L = np.unique(np.asarray(labels), return_inverse=True)[1]; m = int(L.max()) + 1; n = len(L)
    if m <= k:
        return L
    P = sp.csr_matrix((np.ones(n), (np.arange(n), L)), shape=(n, m))
    C = (P.T @ sp.csr_matrix(A) @ P).toarray().astype(np.float64); vol = C.sum(1)
    members = {i: [i] for i in range(m)}; alive = set(range(m))
    while len(alive) > k:
        al = sorted(alive); sub = C[np.ix_(al, al)].copy(); np.fill_diagonal(sub, -1)
        norm = sub / np.sqrt(np.outer(vol[al], vol[al]))
        a, b = np.unravel_index(np.argmax(norm), norm.shape); a, b = al[a], al[b]
        C[a, :] += C[b, :]; C[:, a] += C[:, b]; vol[a] += vol[b]
        members[a] += members.pop(b); alive.discard(b); C[b, :] = 0; C[:, b] = 0
    lut = np.empty(m, int)
    for new, (_, mem) in enumerate(sorted(members.items())):
        lut[mem] = new
    return lut[L]


def leiden_psc(A, k, factor=FACTOR, seed=0):
    """Leiden at ceil(factor * k) clusters (bisection, Leiden seed `seed`), then merge to k. Returns (labels, resolution)."""
    pre, res = leiden_at_k(A, int(np.ceil(factor * k)), seed=seed, hi=6.0)
    return merge_to_k(pre, A, k), res
