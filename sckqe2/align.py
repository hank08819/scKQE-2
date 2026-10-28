"""Quantum alignment: a deterministic donor correction (Methods).

Every gate of the encoding is a rotation R_P(a x) about a fixed Pauli axis P by an angle proportional to the score x.
Rotations about one axis add, R_P(a x) R_P(-a d) = R_P(a (x - d)), so subtracting a donor offset d from a score is the same as
appending a donor-specific rotation about the same axis to each gate. Offsets are estimated inside clusters so that
differences in cell-type composition between donors are kept:
  1. cluster the cells (the KQE-2 clustering at k, Leiden seed 0);
  2. merge every cluster in which one donor holds more than `dominance` of the cells into the cluster with the nearest
     centroid, repeatedly (such a cluster is formed by the donor effect);
  3. inside cluster c, the offset of donor d is (mean of its cells - mean of the cluster) * n_cd / (n_cd + shrink);
  4. subtract each cell's offset, standardize the scores; repeat `iterations` times.
No step draws a random number. Use it when donors are replicates of one condition; when donors differ in biology
(developmental age, disease), their differences are signal and should be kept."""
import numpy as np


def _codes(v):
    return np.unique(v, return_inverse=True)[1]


def merge_donor_clusters(lab, donor, E, dominance=0.8):
    lab = lab.copy()
    while True:
        cl = np.unique(lab)
        if len(cl) < 2:
            return lab
        cent = np.stack([E[lab == c].mean(0) for c in cl])
        share = np.array([np.bincount(_codes(donor[lab == c])).max() / (lab == c).sum() for c in cl])
        bad = np.where(share > dominance)[0]
        if len(bad) == 0:
            return lab
        b = bad[np.argmax(share[bad])]
        d = ((cent - cent[b]) ** 2).sum(1); d[b] = np.inf
        lab[lab == cl[b]] = cl[int(np.argmin(d))]


def donor_offsets(E, lab, donor, shrink=20.0):
    off = np.zeros_like(E, dtype=np.float64); dcode = _codes(donor)
    for c in np.unique(lab):
        ic = np.where(lab == c)[0]; mu_c = E[ic].mean(0); dc = dcode[ic]
        for dv in np.unique(dc):
            j = ic[dc == dv]; n = len(j)
            off[j] = (E[j].mean(0) - mu_c) * (n / (n + shrink))
    return off


def quantum_align(E, donor, cluster_fn, iterations=2, dominance=0.8, shrink=20.0, verbose=False):
    """E: cells x L scores; donor: donor label per cell; cluster_fn(E) -> deterministic cluster labels.
    Returns the aligned, standardized scores (float32)."""
    E = np.asarray(E, np.float64).copy(); donor = np.asarray(donor)
    for it in range(iterations):
        lab = cluster_fn(E); n0 = len(np.unique(lab))
        lab = merge_donor_clusters(lab, donor, E, dominance); n1 = len(np.unique(lab))
        off = donor_offsets(E, lab, donor, shrink); E = E - off; E = (E - E.mean(0)) / E.std(0)
        if verbose:
            print(f"alignment iteration {it + 1}: {n0} clusters, {n0 - n1} donor-dominated merged, mean |offset| {np.abs(off).mean():.4f}")
    return E.astype(np.float32)
