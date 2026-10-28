"""KQE-2 on one complete atlas, as in Table 1: cell filter of the paper, gene-set rule, exact decomposition, entanglement-free
encoding, 15-NN graph, Precision Shrinkage Clustering on whole-transcriptome atlases (Leiden at k on 5,000-HVG atlases), Leiden seed 0.
Writes <out>_clusters.csv.gz and prints the AMI against the annotation. Two runs give bit-identical clusters on the same machine.
Usage: python run_kqe2.py data/fetal.h5ad --label cell_type [--k K] [--donor donor_id] [--out results/fetal]"""
import argparse, os, time, numpy as np, pandas as pd, anndata as ad, sckqe2
from sklearn.metrics import adjusted_mutual_info_score as AMI
ap = argparse.ArgumentParser(); ap.add_argument("h5ad"); ap.add_argument("--label", default="cell_type"); ap.add_argument("--k", type=int)
ap.add_argument("--donor"); ap.add_argument("--out"); ap.add_argument("--save-scores", action="store_true"); a = ap.parse_args()
t0 = time.time(); A = ad.read_h5ad(a.h5ad); X = A.raw.X if A.raw is not None else A.X
y = A.obs[a.label].astype(str).values; keep = (y != "nan") & (y != "unknown")       # unannotated cells and types below 10 cells removed
cls, cnt = np.unique(y[keep], return_counts=True); keep &= np.isin(y, cls[cnt >= 10]); idx = np.where(keep)[0]
X = X[idx]; y = y[idx]; k = a.k or len(np.unique(y)); donors = A.obs[a.donor].astype(str).values[idx] if a.donor else None
m = sckqe2.KQE2(); lab = m.fit_predict(X, k, donors=donors)
out = a.out or os.path.splitext(os.path.basename(a.h5ad))[0]; os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
pd.DataFrame({"cell": np.asarray(A.obs_names)[idx], "annotation": y, "kqe2": lab}).to_csv(out + "_clusters.csv.gz", index=False)
if a.save_scores: np.save(out + "_scores.npy", m.scores_)
print(f"{len(y)} cells, k = {k}; gene set {m.gene_set_used_} (Delta {m.delta_:.2f}); {'PSC' if m._use_psc() else 'Leiden at k'}; "
      f"AMI {AMI(y, lab):.4f}; {time.time() - t0:.0f} s")
