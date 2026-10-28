import json, sys
out = sys.argv[1]
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# scKQE-2 tutorial on Google Colab

**Henry Han**, Data Science and AI Innovation Laboratory, Baylor University (hank08819@gmail.com). scKQE-2 version 1.0.0.

scKQE-2 clusters single-cell RNA-seq data on the whole transcriptome through a reproducible quantum encoding. Each cell becomes a quantum state that holds all of its genes, and cells whose states are close are grouped together. No step is trained and no step draws a random number, so the same counts give the same clusters on every run.

In this notebook you will:
1. connect to a GPU,
2. install scKQE-2,
3. download a public atlas (the human lung cell atlas of Travaglini et al., *Nature* 2020: 65,662 cells, 46 annotated cell types),
4. cluster it, check the result against the annotation and check that a second run is identical,
5. look at the clusters with UMAP and marker genes,
6. run it on your own data.

The whole notebook took 30 minutes on a server GPU (peak memory 9.7 GB); plan for 30 to 60 minutes on a free Colab GPU, whose 12 GB of memory is enough for this atlas.
""")

md("""
## 1. Connect to a GPU

In the menu choose **Runtime → Change runtime type → Hardware accelerator: T4 GPU** (any GPU works), then **Save**.
The cell below checks that a GPU is available. scKQE-2 also runs without a GPU and gives the same clusters; the GPU only makes the quantum encoding and the nearest-neighbor search faster.
""")
code("""
import torch
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU: choose Runtime -> Change runtime type -> T4 GPU, or continue on the CPU (slower, same result).")
""")

md("""
## 2. Install scKQE-2

Before the public release, upload the file `scKQE2-1.0.0.zip` when the cell asks for it (it is in the `dist` folder of the package).
After the release, the line `!pip install scKQE2` is enough.
""")
code("""
!pip -q install scanpy anndata umap-learn igraph leidenalg
import importlib.util
if importlib.util.find_spec("sckqe2") is None:
    try:
        !pip -q install scKQE2
    except Exception:
        pass
if importlib.util.find_spec("sckqe2") is None:
    from google.colab import files
    print("Upload scKQE2-1.0.0.zip")
    up = files.upload()
    for name in up:
        !pip -q install --no-deps "{name}"
import sckqe2
print("scKQE-2 is installed")
""")

md("""
## 3. Download a public atlas

The human lung cell atlas (Travaglini et al., *Nature* 587, 619-625, 2020; doi:10.1038/s41586-020-2922-4) is distributed by CZ CELLxGENE Discover.
The cell asks CELLxGENE for the current download link of the dataset and saves it as `lung.h5ad` (about 600 MB).
""")
code("""
import json, urllib.request, os
DATASET_ID = "8c42cfd0-0b0a-46d5-910c-fc833d83c45e"   # Krasnow Lab Human Lung Cell Atlas, 10X
info = json.load(urllib.request.urlopen(f"https://api.cellxgene.cziscience.com/curation/v1/datasets/{DATASET_ID}/versions"))
url = [a["url"] for a in info[0]["assets"] if a["filetype"] == "H5AD"][0]
print(info[0]["title"], "-", info[0]["cell_count"], "cells")
if not os.path.exists("lung.h5ad"):
    !wget -q -O lung.h5ad "{url}"
!ls -lh lung.h5ad
""")

md("""
## 4. Load the counts

scKQE-2 reads **raw counts**. It takes them from `adata.raw.X` when it exists (as in CELLxGENE files), otherwise from `adata.X`.
We keep annotated cells and cell types with at least 10 cells. The number of annotated types sets the number of clusters `k`; the labels are used for nothing else.
""")
code("""
import time
import numpy as np
import anndata as ad
import scanpy as sc

adata = ad.read_h5ad("lung.h5ad")
y = adata.obs["cell_type"].astype(str)
keep = ~y.isin(["unknown", "nan"])
counts = y[keep].value_counts()
keep &= y.isin(counts.index[counts >= 10])
adata = adata[keep.values].copy()
k = adata.obs["cell_type"].nunique()
print(f"{adata.n_obs} cells, {adata.n_vars} genes, {k} annotated cell types")
print("smallest types:"); print(adata.obs["cell_type"].value_counts().tail(5))
""")

md("""
## 5. Choose the gene set

scKQE-2 uses the whole transcriptome by default. A label-free rule checks whether 5,000 highly variable genes would be better: it measures the data entropy of the signal directions on 10,000 cells and compares it with log2(k).
""")
code("""
X = adata.raw.X if adata.raw is not None else adata.X
gene_set, delta, h_pa, n_signal = sckqe2.choose_gene_set(X, k)
print(f"Delta = {delta:.2f} (data entropy {h_pa:.2f} bits, {n_signal} signal directions) -> gene set: {gene_set}")
""")

md("""
## 6. Cluster with scKQE-2

The counts get one fixed transformation, with nothing to choose: log(1 + count) for every gene, multiplied by the gene's IDF weight (genes seen in few cells, such as rare-type markers, weigh more), and each cell scaled to unit length; no gene scaling, covariate regression or batch correction.\n\nThe steps inside are: IDF-weighted log counts of the whole transcriptome; an exact decomposition into 132 directions (no random seed); a quantum encoding of each direction on its own qubit, in blocks of 6 qubits with no entangling gate; a 15-nearest-neighbor graph; and Leiden at `k` clusters.
""")
code("""
t0 = time.time()
sckqe2.kqe2(adata, n_clusters=k, gene_set=gene_set)
print(f"{adata.obs['kqe2'].nunique()} clusters in {time.time() - t0:.0f} s; quantum features {adata.obsm['X_kqe2'].shape}")
""")

md("""
## 7. Compare with the annotation

The adjusted mutual information (AMI) measures how well the clusters agree with the annotated cell types (1 = perfect agreement).
""")
code("""
from sklearn.metrics import adjusted_mutual_info_score
print(f"AMI with the annotation: {adjusted_mutual_info_score(adata.obs['cell_type'], adata.obs['kqe2']):.4f}")
""")

md("""
## 8. Check reproducibility

Run scKQE-2 again. Both checks print `True`: the quantum features are identical bit for bit and every cell lands in the same cluster.
""")
code("""
first_features, first_clusters = adata.obsm["X_kqe2"].copy(), adata.obs["kqe2"].values.copy()
sckqe2.kqe2(adata, n_clusters=k, gene_set=gene_set)
print("features identical:", np.array_equal(first_features, adata.obsm["X_kqe2"]))
print("clusters identical:", np.array_equal(first_clusters, adata.obs["kqe2"].values))
""")

md("""
## 9. UMAP and marker genes

The quantum features plug into any scanpy workflow. Marker genes are found on the whole transcriptome, with gene symbols.
""")
code("""
sc.pp.neighbors(adata, use_rep="X_kqe2", n_neighbors=15, random_state=0)
sc.tl.umap(adata, random_state=0)
sc.pl.umap(adata, color=["kqe2", "cell_type"], legend_loc="on data", legend_fontsize=6, frameon=False, ncols=1)
""")
code("""
lognorm = ad.AnnData(X.copy(), obs=adata.obs[["kqe2"]].copy())
lognorm.var_names = adata.raw.var["feature_name"].astype(str).values if adata.raw is not None and "feature_name" in adata.raw.var else adata.var["feature_name"].astype(str).values
lognorm.var_names_make_unique()
sc.pp.normalize_total(lognorm, target_sum=1e4); sc.pp.log1p(lognorm)
sc.tl.rank_genes_groups(lognorm, "kqe2", method="wilcoxon", n_genes=5)
top = sc.get.rank_genes_groups_df(lognorm, group=None).groupby("group").head(3)
major = adata.obs.groupby("kqe2", observed=True)["cell_type"].agg(lambda s: s.value_counts().index[0])
for g, d in list(top.groupby("group", observed=True))[:10]:
    print(f"cluster {g} (mostly {major[int(g)] if int(g) in major.index else major[g]}): {', '.join(d['names'])}")
""")

md("""
## 10. Donor correction (optional)

When donors are replicates of one condition, scKQE-2 can remove donor offsets deterministically (quantum alignment). The lung atlas comes from three donors with normal lungs, so the correction applies here.
**Do not use it** when donors differ in biology that you want to keep, such as developmental age or disease status.
""")
code("""
sckqe2.kqe2(adata, n_clusters=k, gene_set=gene_set, donor_key="donor_id", key_added="kqe2_aligned")
print(f"AMI after quantum alignment: {adjusted_mutual_info_score(adata.obs['cell_type'], adata.obs['kqe2_aligned']):.4f}")
""")

md("""
## 11. Your own data

Put an `.h5ad` file with raw counts in your Google Drive, then run the cell below with your file name and label column.
Without an annotation, set `k` from biology or try several values; the result for each `k` is reproducible.
Free Colab sessions have about 12 GB of memory, which fits up to a few hundred thousand cells; larger atlases need a machine with more memory.
""")
code("""
# from google.colab import drive
# drive.mount("/content/drive")
# mine = ad.read_h5ad("/content/drive/MyDrive/my_atlas.h5ad")
# k_mine = 20                                         # or mine.obs["cell_type"].nunique()
# Xm = mine.raw.X if mine.raw is not None else mine.X
# gs = sckqe2.choose_gene_set(Xm, k_mine)[0]
# sckqe2.kqe2(mine, n_clusters=k_mine, gene_set=gs)
# mine.obs[["kqe2"]].to_csv("/content/drive/MyDrive/my_atlas_kqe2_clusters.csv")
""")

md("""
## Cite

If you use scKQE-2, please cite: H. Han, *Reproducible quantum clustering of atlas-scale single-cell data* (2026).
The lung atlas: K. J. Travaglini et al., A molecular cell atlas of the human lung from single-cell RNA sequencing, *Nature* 587, 619-625 (2020).
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(out, "w"), indent=1)
print("written", len(cells), "cells")
