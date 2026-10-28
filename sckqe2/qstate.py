"""Exact quantum-state geometry of a complete atlas (Theorem 4; Fig. 3).

Every cell is the product state of its 132 qubits, |psi(x)> = prod_q V(x_q)|0>, described exactly by its Bloch vectors r_q.
The fidelity of two cells is F = prod_q (1 + r_q . r'_q) / 2 and the state distance is -log F.

Exact nearest states for millions of cells (certified search). Candidates are the M cells nearest in Euclidean distance
between the Bloch-vector features f (matrix products on the GPU); their exact -log F is computed and the K smallest are kept.
By Theorem 4(c), every cell outside the candidate list has -log F >= ||f - f'||^2 / 4 >= e_M / 4, where e_M is the M-th
smallest squared feature distance. When e_M / 4 is at least the K-th smallest exact distance among the candidates, the K
neighbors are exact; cells without this certificate are searched exhaustively. No approximation is made.
On the Siletti atlas (888,263 nuclei) 117 nuclei needed the exhaustive search (100 s on one GPU); on the complete mouse
embryo atlas (11,441,407 nuclei), 104 (3.1 h on one GPU).
"""
import numpy as np
from .encode import bloch_vectors


def state_distance(Ra, Rb):
    """-log F between product states given by Bloch vectors (..., L, 3)."""
    dot = (np.asarray(Ra, np.float64) * np.asarray(Rb, np.float64)).sum(-1)
    return -np.log(np.clip((1 + dot) / 2, 1e-12, None)).sum(-1)


def bloch_chunked(E, s=0.19, chunk=500000):
    """Bloch vectors (cells, L, 3) as float32, computed in chunks (float64 inside) to bound memory on large atlases."""
    E = np.asarray(E); R = np.empty(E.shape + (3,), np.float32)
    for i in range(0, len(E), chunk):
        R[i:i + chunk] = bloch_vectors(E[i:i + chunk], s)
    return R


def certified_knn(R, k=15, candidates=200, chunk=128, block=1_500_000, device=None):
    """Exact k nearest states (self included as neighbor 0) under -log F for Bloch vectors R (cells, L, 3).
    Returns (I, D, certified): neighbor indices, -log F distances, and whether the Theorem 4 certificate held for each cell
    (cells without it were searched exhaustively, so all neighbors are exact either way). Needs torch; uses the GPU when present."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n = len(R); M = min(candidates, n); k = min(k, n)
    T = torch.as_tensor(np.ascontiguousarray(R, np.float32), device=dev); F = T.reshape(n, -1); sq = (F * F).sum(1)
    I = np.empty((n, k), np.int64); D = np.empty((n, k), np.float32); cert = np.zeros(n, bool)
    for s0 in range(0, n, chunk):
        e = min(s0 + chunk, n); rows = torch.arange(s0, e, device=dev)
        d2 = sq[s0:e, None] - 2 * F[s0:e] @ F.T + sq[None, :]
        v, c = torch.topk(d2, M, dim=1, largest=False); del d2
        dot = (T[rows][:, None] * T[c]).sum(-1)
        nlf = -torch.log(((1 + dot) * 0.5).clamp(min=1e-12)).sum(-1)
        nv, ni = torch.topk(nlf, k, dim=1, largest=False)
        cert[s0:e] = (M == n) | (v[:, -1].clamp(min=0) / 4 >= nv[:, -1]).cpu().numpy()
        I[s0:e] = torch.gather(c, 1, ni).cpu().numpy(); D[s0:e] = nv.clamp(min=0).cpu().numpy()
    Tx, Ty, Tz = T[..., 0], T[..., 1], T[..., 2]
    for r in np.where(~cert)[0]:
        bv, bi = [], []
        for c0 in range(0, n, block):
            dot = Tx[r][None, :] * Tx[c0:c0 + block]; dot.addcmul_(Ty[r][None, :], Ty[c0:c0 + block]); dot.addcmul_(Tz[r][None, :], Tz[c0:c0 + block])
            nl = -torch.log(dot.add_(1).mul_(0.5).clamp_(min=1e-12)).sum(-1)
            v, i = torch.topk(nl, min(k, len(nl)), largest=False); bv.append(v); bi.append(i + c0)
        v = torch.cat(bv); i = torch.cat(bi); o = torch.topk(v, k, largest=False).indices
        I[r] = i[o].cpu().numpy(); D[r] = v[o].clamp(min=0).cpu().numpy()
    return I, D, cert


def state_knn(E, s=0.19, k=15, **kw):
    """Certified exact nearest states from whitened scores E (cells x L)."""
    return certified_knn(bloch_chunked(E, s), k=k, **kw)


def state_map(I, D, seed=0, min_dist=0.3, low_memory=True):
    """2-D map (UMAP) of the exact nearest-state graph, as in Fig. 3b,e."""
    import umap
    n, k = I.shape
    return umap.UMAP(n_neighbors=k, min_dist=min_dist, random_state=seed, precomputed_knn=(I, np.sqrt(D)),
                     low_memory=low_memory).fit_transform(np.zeros((n, 2), np.float32))


def neighbor_purity(I, labels):
    """Share of each cell's nearest states (neighbor 0, the cell itself, excluded) that carry its label; mean over cells."""
    labels = np.asarray(labels)
    return float((labels[I[:, 1:]] == labels[:, None]).mean())
