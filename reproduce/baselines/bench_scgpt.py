"""scGPT zero-shot cell embeddings (whole-human checkpoint) on the shared subsample -> emb_scgpt.npy. Human datasets only."""
import os, sys, time, json, numpy as np, anndata as ad
from scgpt.tasks import embed_data
d = sys.argv[1]; MODEL = os.environ.get("SCGPT_MODEL", "scgpt_model/whole-human")  # path to the scGPT whole-human checkpoint; A = ad.read_h5ad(os.path.join(d, "sub.h5ad"))
if os.path.exists(os.path.join(d, "emb_scgpt.npy")): sys.exit(0)
A.var["gene_name"] = [str(x).upper() for x in A.var["gene_name"].values]; t0 = time.time()
E = embed_data(A, MODEL, gene_col="gene_name", use_fast_transformer=False, device="cuda", batch_size=128, return_new_adata=False)
np.save(os.path.join(d, "emb_scgpt.npy"), np.asarray(E.obsm["X_scGPT"], np.float32)); json.dump({"scgpt": time.time() - t0}, open(os.path.join(d, "time_scgpt.json"), "w"))
print("scgpt secs", time.time() - t0, flush=True)
