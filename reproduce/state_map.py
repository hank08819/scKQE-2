"""Fig. 3b-f: exact nearest quantum states of every cell (Theorem 4 certificate) and the map built from them.
Input: whitened scores saved by run_kqe2.py --save-scores and its clusters file. Writes <out>_state.npz and <out>_state.png.
Usage: python state_map.py results/siletti_scores.npy results/siletti_clusters.csv.gz --out results/siletti"""
import argparse, time, numpy as np, pandas as pd, sckqe2
ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("clusters"); ap.add_argument("--out", default="state"); ap.add_argument("--k", type=int, default=15)
a = ap.parse_args(); t0 = time.time()
E = np.load(a.scores); C = pd.read_csv(a.clusters); I, D, cert = sckqe2.state_knn(E, k=a.k)
print(f"{len(E)} cells: certified by Theorem 4 {cert.sum()} ({100 * cert.mean():.3f}%), exhaustive {int((~cert).sum())}; {time.time() - t0:.0f} s")
print(f"share of the {a.k} nearest states in the own KQE-2 cluster {sckqe2.neighbor_purity(I, C.kqe2.values):.4f}; "
      f"in the own annotated type {sckqe2.neighbor_purity(I, C.annotation.values):.4f}; median neighbour fidelity {np.exp(-np.median(D[:, 1:])):.3f}")
U = sckqe2.state_map(I, D); np.savez_compressed(a.out + "_state.npz", I=I, D=D, certified=cert, umap=U)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
codes = pd.factorize(C.kqe2.values)[0]; fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(U[:, 0], U[:, 1], s=0.05, c=codes, cmap="tab20", linewidths=0, rasterized=True); ax.set_axis_off(); fig.savefig(a.out + "_state.png", dpi=200)
