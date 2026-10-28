"""scKQE-2 tutorial script (version 2.0): every code block of docs/tutorial.md, run in order on one atlas.
Usage: python tutorial.py atlas.h5ad [label column] [donor column]
The outputs in docs/tutorial.md come from: python tutorial.py lung.h5ad cell_type donor_id
(Krasnow Lab Human Lung Cell Atlas, 10X; CELLxGENE dataset 8c42cfd0-0b0a-46d5-910c-fc833d83c45e)."""
import sys, time
import numpy as np
import anndata as ad
import scanpy as sc
import sckqe2
from sklearn.metrics import adjusted_mutual_info_score

path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else "cell_type"; donor = sys.argv[3] if len(sys.argv) > 3 else None

# 1. Load raw counts; keep annotated cells and types with at least 10 cells
adata = ad.read_h5ad(path)
y = adata.obs[label].astype(str)
keep = ~y.isin(["unknown", "nan"])
counts = y[keep].value_counts()
keep &= y.isin(counts.index[counts >= 10])
adata = adata[keep.values].copy()
k = adata.obs[label].nunique()
print(f"[1] {adata.n_obs} cells, {adata.n_vars} genes, {k} annotated types")

# 2. Cluster: the gene-set rule, the exact decomposition, the entanglement-free encoding and the clustering of the paper
t0 = time.time()
model = sckqe2.kqe2(adata, n_clusters=k)
info = adata.uns["kqe2"]
print(f"[2] gene-set rule: Delta = {info['delta']:.2f} -> {info['gene_set_used']}; clustering: "
      f"{'Precision Shrinkage Clustering' if info['psc'] else 'Leiden at k'}; {adata.obs['kqe2'].nunique()} clusters in "
      f"{time.time() - t0:.0f} s; features {adata.obsm['X_kqe2'].shape}")

# 3. Compare with the annotation (labels are used only here and to set k)
print(f"[3] AMI with '{label}': {adjusted_mutual_info_score(adata.obs[label], adata.obs['kqe2']):.4f}")

# 4. Check reproducibility: a second run gives bit-identical features and the same clusters
first_features, first_clusters = adata.obsm["X_kqe2"].copy(), adata.obs["kqe2"].values.copy()
sckqe2.kqe2(adata, n_clusters=k)
print(f"[4] features identical: {np.array_equal(first_features, adata.obsm['X_kqe2'])}; "
      f"clusters identical: {np.array_equal(first_clusters, adata.obs['kqe2'].values)}")

# 5. Exact quantum-state geometry: nearest quantum states certified by Theorem 4, and the map built from them
t0 = time.time()
I, D, certified = model.certified_state_neighbors(k=15)
print(f"[5] certified by Theorem 4: {certified.sum()} of {len(certified)} cells; the rest searched exhaustively; {time.time() - t0:.0f} s")
print(f"    share of the 15 nearest states in the own cluster: {sckqe2.neighbor_purity(I, adata.obs['kqe2'].values):.4f}; "
      f"median fidelity of nearest states: {np.exp(-np.median(D[:, 1:])):.3f}")
adata.obsm["X_state_map"] = sckqe2.state_map(I, D)

# 6. Downstream analysis with scanpy: markers per cluster on the whole transcriptome
lognorm = adata.copy(); lognorm.X = (adata.raw.X if adata.raw is not None else adata.X).copy(); lognorm.raw = None
if "feature_name" in lognorm.var.columns:                       # show gene symbols instead of Ensembl IDs
    lognorm.var_names = lognorm.var["feature_name"].astype(str).values; lognorm.var_names_make_unique()
sc.pp.normalize_total(lognorm, target_sum=1e4); sc.pp.log1p(lognorm)
sc.tl.rank_genes_groups(lognorm, "kqe2", method="wilcoxon", n_genes=5)
top = sc.get.rank_genes_groups_df(lognorm, group=None).groupby("group").head(3)
major = adata.obs.groupby("kqe2", observed=True)[label].agg(lambda s: s.value_counts().index[0])
print("[6] top markers of the three largest clusters:")
for g in adata.obs["kqe2"].value_counts().index[:3]:
    print(f"    cluster {g} ({major[g]}): {', '.join(map(str, top[top.group == g]['names']))}")

# 7. Optional: deterministic donor correction (quantum alignment)
if donor:
    sckqe2.kqe2(adata, n_clusters=k, donor_key=donor, key_added="kqe2_aligned")
    print(f"[7] {adata.obs[donor].nunique()} donors; AMI after quantum alignment: "
          f"{adjusted_mutual_info_score(adata.obs[label], adata.obs['kqe2_aligned']):.4f}")

# 8. Optional: the circuit of one block on quantum hardware, and the exact values its measurements are checked against
try:
    from sckqe2.hardware import block_circuit, exact_expectations, simulate
    x = model.scores_[0, :6]
    print(f"[8] circuit of one block: {block_circuit(x).count_ops()}; largest gap between the noise-free circuit and the "
          f"closed form: {np.abs(simulate(x) - exact_expectations(x)).max():.1e}")
except ImportError:
    print("[8] install qiskit (pip install 'scKQE2[hardware]') for the hardware section")
