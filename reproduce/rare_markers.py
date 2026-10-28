"""Rare-type check (Results, Rare embryonic types): F1 of the KQE-2 cluster holding most of each type, and canonical markers
(share of nuclei with a nonzero count inside vs outside the cluster; 3,000 nuclei each, seed 0) plus the top 10 Wilcoxon markers.
Usage: python rare_markers.py data/mouse_embryo_complete.h5ad results/mouse_embryo_complete_clusters.csv.gz"""
import sys, numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp
CAN = {"notochordal cell": ["T", "Noto", "Shh"], "Purkinje cell": ["Pcp2", "Calb1", "Car8"], "endocardial cell": ["Npr3", "Nfatc1"],
       "amacrine cell": ["Gad1", "Slc6a9", "Tfap2b"], "epithelial cell of amnion": ["Isl1", "Postn"]}
A = ad.read_h5ad(sys.argv[1], backed="r"); C = pd.read_csv(sys.argv[2]); pos = pd.Series(np.arange(A.n_obs), index=A.obs_names).reindex(C.cell).values
y = C.annotation.astype(str).values; L = C.kqe2.values; names = A.var["feature_name"].astype(str).values if "feature_name" in A.var else A.var_names.values; rs = np.random.RandomState(0)
for t, genes in CAN.items():
    m = y == t
    if not m.any(): continue
    c = pd.Series(L[m]).value_counts().index[0]; M = L == c; f1 = 2 * (M & m).sum() / (M.sum() + m.sum())
    i1 = rs.choice(np.where(M)[0], min(3000, M.sum()), replace=False); i0 = rs.choice(np.where(~M)[0], 3000, replace=False); idx = np.sort(np.r_[i1, i0])
    B = ad.AnnData(sp.csr_matrix(A.X[np.sort(pos[idx])]).astype(np.float32)); B.var_names = pd.Index(names); B.var_names_make_unique()
    sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B); B.obs["g"] = pd.Categorical(np.where(np.isin(idx, i1), "in", "out"))
    can = " | ".join(f"{g} {100 * (B[B.obs.g == 'in', g].X.toarray() > 0).mean():.0f}% vs {100 * (B[B.obs.g == 'out', g].X.toarray() > 0).mean():.0f}%" for g in genes if g in B.var_names)
    sc.tl.rank_genes_groups(B, "g", groups=["in"], reference="out", method="wilcoxon", n_genes=10)
    print(f"{t}: {m.sum()} nuclei, F1 {f1:.3f}; {can}; top: {', '.join(sc.get.rank_genes_groups_df(B, 'in').names)}")
