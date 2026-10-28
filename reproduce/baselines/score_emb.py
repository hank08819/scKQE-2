"""Score saved embeddings with the unified protocol of the paper: 15-NN exact graph (GPU), UMAP fuzzy weights,
Leiden bisected to k (3 seeds, mean AMI) and best AMI over resolutions 0.02..1.0 step 0.02 (seed 0).
Usage: python score_emb.py <dir> [emb names...]  -> <dir>/scores.csv (appends)"""
import os, sys, time, glob, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc, torch
from umap.umap_ import fuzzy_simplicial_set
from sklearn.metrics import adjusted_mutual_info_score as AMI
d = sys.argv[1]; names = sys.argv[2:] or [os.path.basename(f)[4:].rsplit(".", 1)[0] for f in sorted(glob.glob(os.path.join(d, "emb_*")))]
y = pd.factorize(pd.read_csv(os.path.join(d, "labels.tsv"), header=None)[0].astype(str).values)[0]; k = len(np.unique(y)); n = len(y)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); GRID = np.round(np.arange(0.02, 1.001, 0.02), 2)
def knn(F, kk=15, ch=int(os.environ.get("KNN_CH", 2048))):
    X = torch.as_tensor(F, device=DEV); sq = (X * X).sum(1); I = np.empty((n, kk), np.int64); D = np.empty((n, kk), np.float32)
    for s0 in range(0, n, ch):
        e = min(s0 + ch, n); rows = torch.arange(s0, e, device=DEV); d2 = sq[s0:e, None] - 2 * X[s0:e] @ X.T + sq[None, :]; d2[torch.arange(e - s0, device=DEV), rows] = float("inf")
        v, i = torch.topk(d2, kk - 1, dim=1, largest=False); I[s0:e, 0] = rows.cpu().numpy(); D[s0:e, 0] = 0; I[s0:e, 1:] = i.cpu().numpy(); D[s0:e, 1:] = v.clamp(min=0).sqrt().cpu().numpy()
    return I, D
rows = []
for nm in names:
    f = [p for p in glob.glob(os.path.join(d, f"emb_{nm}.*"))][0]; F = np.load(f) if f.endswith(".npy") else np.loadtxt(f, delimiter=",")
    F = np.ascontiguousarray(F, np.float32); assert len(F) == n, (nm, F.shape, n); t0 = time.time(); I, D = knn(F)
    G, _, _ = fuzzy_simplicial_set(sp.coo_matrix((n, 1)), 15, None, "euclidean", knn_indices=I, knn_dists=D, set_op_mix_ratio=1.0, local_connectivity=1.0); G = G.tocsr()
    A = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(n)])); atk = []; ks = []
    def run(res, seed):
        sc.tl.leiden(A, resolution=res, key_added="L", flavor="igraph", n_iterations=2, directed=False, adjacency=G, random_state=seed); L = A.obs["L"].astype(int).values; return len(np.unique(L)), AMI(y, L)
    for seed in (0, 1, 2):
        lo, hi, best = 0.02, 3.0, None
        for _ in range(12):
            res = (lo + hi) / 2; kk_, am = run(res, seed)
            if best is None or abs(kk_ - k) < abs(best[0] - k): best = (kk_, am)
            if kk_ == k: break
            if kk_ < k: lo = res
            else: hi = res
        atk.append(best[1]); ks.append(best[0])
    sw = [(r, *run(r, 0)) for r in GRID]; bst = max(sw, key=lambda t: t[2])
    r = dict(method=nm, dim=F.shape[1], at_k=float(np.mean(atk)), at_k_seeds=",".join(f"{v:.4f}" for v in atk), k_found=",".join(map(str, ks)), sweep_best=bst[2], sweep_k=bst[1], secs=time.time() - t0); rows.append(r)
    print(f"   {nm:12s} dim {F.shape[1]:4d}  at_k {r['at_k']:.4f} ({r['at_k_seeds']}; k {r['k_found']})  sweep {bst[2]:.4f} (k={bst[1]})", flush=True)
out = os.path.join(d, "scores.csv"); old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(); pd.concat([old[~old.method.isin(names)] if len(old) else old, pd.DataFrame(rows)]).to_csv(out, index=False)
