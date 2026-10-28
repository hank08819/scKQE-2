"""Embeddings of the shared subsample for Python baselines: scanpy (HVG2000, PCA50) and scVI (HVG2000, n_latent 30).
Saves <dir>/emb_<method>.npy. Usage: python bench_python.py <dir> [methods=scanpy,scvi]"""
import os, sys, time, json, numpy as np, scanpy as sc, anndata as ad
d = sys.argv[1]; methods = (sys.argv[2] if len(sys.argv) > 2 else "scanpy,scvi").split(","); A = ad.read_h5ad(os.path.join(d, "sub.h5ad")); tim = {}
if "scanpy" in methods:
    t0 = time.time(); B = A.copy(); sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B); sc.pp.highly_variable_genes(B, n_top_genes=2000)
    B = B[:, B.var.highly_variable].copy(); sc.pp.scale(B, max_value=10); sc.tl.pca(B, n_comps=50, random_state=0)
    np.save(os.path.join(d, "emb_scanpy.npy"), np.asarray(B.obsm["X_pca"], np.float32)); tim["scanpy"] = time.time() - t0; del B
if "scvi" in methods:
    import scvi, torch; scvi.settings.seed = 0; t0 = time.time(); B = A.copy(); B.layers["counts"] = B.X.copy()
    C = B.copy(); sc.pp.normalize_total(C, target_sum=1e4); sc.pp.log1p(C); sc.pp.highly_variable_genes(C, n_top_genes=2000); B = B[:, C.var.highly_variable].copy(); del C
    scvi.model.SCVI.setup_anndata(B, layer="counts"); m = scvi.model.SCVI(B, n_latent=30); m.train(accelerator="gpu", devices=1, early_stopping=True)
    np.save(os.path.join(d, "emb_scvi.npy"), np.asarray(m.get_latent_representation(), np.float32)); tim["scvi"] = time.time() - t0
json.dump(tim, open(os.path.join(d, "time_python.json"), "w")); print("times", tim, flush=True)
