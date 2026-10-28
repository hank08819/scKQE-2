"""Command line: sckqe2 atlas.h5ad --k 13 [--label cell_type] [--donor donor_id] [--gene-set auto|all|hvg5000] [--psc auto|on|off] [--out out.csv]"""
import argparse
import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sckqe2", description="scKQE-2: reproducible quantum clustering of single-cell data")
    ap.add_argument("h5ad"); ap.add_argument("--k", type=int, help="number of clusters (default: number of types in --label)")
    ap.add_argument("--label", help="obs column with annotated types (only to set k and to report AMI)")
    ap.add_argument("--donor", help="obs column with donors: enables quantum alignment")
    ap.add_argument("--gene-set", default="auto", choices=["auto", "all", "hvg5000"])
    ap.add_argument("--psc", default="auto", choices=["auto", "on", "off"], help="Precision Shrinkage Clustering (auto: on for whole-transcriptome atlases)")
    ap.add_argument("--backend", default="statevector", choices=["statevector", "closed_form"])
    ap.add_argument("--out", default="sckqe2_clusters.csv")
    a = ap.parse_args(argv)
    import anndata as ad
    from .api import KQE2
    from .rule import choose_gene_set
    A = ad.read_h5ad(a.h5ad); X = A.raw.X if A.raw is not None else A.X
    keep = np.ones(A.n_obs, bool); y = None
    if a.label:
        y = A.obs[a.label].astype(str).values; keep = (y != "nan") & (y != "unknown")
        cls, cnt = np.unique(y[keep], return_counts=True); keep &= np.isin(y, cls[cnt >= 10])
    X = X[keep]; k = a.k or len(np.unique(y[keep]))
    psc = {"auto": "auto", "on": True, "off": False}[a.psc]
    model = KQE2(gene_set=a.gene_set, psc=psc, backend=a.backend)
    donors = A.obs[a.donor].astype(str).values[keep] if a.donor else None
    lab = model.fit_predict(X, k, donors=donors)
    if model.delta_ is not None:
        print(f"gene-set rule: Delta = {model.delta_:.2f} (h_PA {model.h_signal_:.2f}, {model.n_signal_} signal directions) -> {model.gene_set_used_}")
    print(f"clustering: {'Precision Shrinkage Clustering' if model._use_psc() else 'Leiden at k'}")
    out = pd.DataFrame({"cell": np.asarray(A.obs_names)[keep], "cluster": lab})
    if y is not None:
        from sklearn.metrics import adjusted_mutual_info_score as AMI
        out["annotation"] = y[keep]; print(f"AMI with '{a.label}': {AMI(y[keep], lab):.4f}")
    out.to_csv(a.out, index=False); print(f"{len(out)} cells, {len(np.unique(lab))} clusters -> {a.out}")


if __name__ == "__main__":
    main()
