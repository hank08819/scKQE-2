"""High-level interface: the KQE2 estimator and scanpy-style functions on AnnData."""
import numpy as np
from .preprocess import to_csr32, idf_matrix_inplace, hvg_index
from .decompose import exact_scores
from .encode import encode
from .cluster import knn_graph, leiden_at_k
from .align import quantum_align
from .psc import leiden_psc
from .rule import choose_gene_set


class KQE2:
    """scKQE-2: cluster cells on the whole transcriptome through a reproducible quantum encoding.

    Steps (the configuration of the paper, Table 1):
      1. gene-set rule (Delta = h_PA - log2 k, locked threshold 1.15): whole transcriptome with IDF, or 5,000 HVGs;
      2. exact decomposition (ARPACK, fixed start vector, no random seed), 132 whitened scores;
      3. quantum encoding without entanglement (6 qubits per block, 3 layers, s = 0.19), 726 exact features;
      4. 15-nearest-neighbor graph with UMAP fuzzy weights;
      5. Precision Shrinkage Clustering on whole-transcriptome atlases, Leiden at k on 5,000-HVG atlases (psc="auto").
    Optional deterministic donor correction (quantum alignment). gene_set="all"/"hvg5000" and psc=True/False override the rules.

    >>> model = KQE2()
    >>> labels = model.fit_predict(counts, n_clusters=13)
    """

    def __init__(self, n_directions=132, gene_set="auto", psc="auto", qubits=6, scale=0.19, n_neighbors=15, tol=1e-6,
                 backend="statevector", leiden_seed=0, device=None):
        self.n_directions, self.gene_set, self.psc, self.qubits, self.scale = n_directions, gene_set, psc, qubits, scale
        self.n_neighbors, self.tol, self.backend, self.leiden_seed, self.device = n_neighbors, tol, backend, leiden_seed, device

    def scores(self, counts):
        """Exact expression scores (cells x n_directions). gene_set: 'all' (IDF) or 'hvg5000' (no IDF, depth 1e4)."""
        X = to_csr32(counts)
        if self.gene_set == "auto":
            raise ValueError("gene_set='auto' needs the number of clusters: call fit_predict, or resolve it with sckqe2.rule.choose_gene_set")
        if self.gene_set == "hvg5000":
            X = to_csr32(X[:, hvg_index(X, 5000)]); Xi = idf_matrix_inplace(X.copy(), library_size=True, idf=False)
        elif self.gene_set == "all":
            Xi = idf_matrix_inplace(X.copy())
        else:
            raise ValueError("gene_set must be 'all' or 'hvg5000' (see sckqe2.rule.choose_gene_set)")
        self.gene_set_used_ = getattr(self, "gene_set_used_", None) or self.gene_set
        self.scores_ = exact_scores(Xi, self.n_directions, whiten=True, tol=self.tol)
        return self.scores_

    def features(self, E):
        return encode(E, self.scale, self.qubits, backend=self.backend, device=self.device)

    def fit_transform(self, counts):
        self.features_ = self.features(self.scores(counts))
        return self.features_

    def _use_psc(self):
        return (self.gene_set_used_ == "all") if self.psc == "auto" else bool(self.psc)

    def cluster(self, F, n_clusters):
        self.graph_ = knn_graph(F, self.n_neighbors, device=self.device)
        if self._use_psc():
            self.labels_, self.resolution_ = leiden_psc(self.graph_, n_clusters, seed=self.leiden_seed)
        else:
            self.labels_, self.resolution_ = leiden_at_k(self.graph_, n_clusters, seed=self.leiden_seed)
        return self.labels_

    def fit_predict(self, counts, n_clusters, donors=None, align_iterations=2):
        """Cluster into n_clusters. With donors (one label per cell), quantum alignment corrects donor offsets first."""
        requested = self.gene_set
        if requested == "auto":
            self.gene_set_used_, self.delta_, self.h_signal_, self.n_signal_ = choose_gene_set(counts, n_clusters)
        else:
            self.gene_set_used_, self.delta_ = requested, None
        self.gene_set = self.gene_set_used_
        try:
            E = self.scores(counts)
        finally:
            self.gene_set = requested
        if donors is not None:
            E = quantum_align(E, donors, lambda S: self.cluster(self.features(S), n_clusters), iterations=align_iterations)
            self.aligned_scores_ = E
        self.features_ = self.features(E)
        return self.cluster(self.features_, n_clusters)

    def certified_state_neighbors(self, k=15, **kw):
        """Exact nearest quantum states of every cell (Theorem 4 certificate; see sckqe2.qstate.certified_knn)."""
        from .qstate import state_knn
        return state_knn(self.scores_, s=self.scale, k=k, **kw)


def kqe2(adata, n_clusters, donor_key=None, layer=None, use_raw=True, key_added="kqe2", **kwargs):
    """scanpy-style: writes adata.obsm['X_kqe2'] (quantum features) and adata.obs[key_added] (clusters). Raw counts are
    taken from adata.raw.X when present and use_raw=True, else from adata.layers[layer] or adata.X."""
    X = adata.layers[layer] if layer else (adata.raw.X if (use_raw and adata.raw is not None) else adata.X)
    model = KQE2(**kwargs)
    donors = adata.obs[donor_key].astype(str).values if donor_key else None
    lab = model.fit_predict(X, n_clusters, donors=donors)
    adata.obsm["X_" + key_added] = model.features_
    adata.obs[key_added] = np.asarray(lab).astype(str)
    adata.uns[key_added] = dict(gene_set_used=model.gene_set_used_, psc=bool(model._use_psc()), delta=model.delta_,
                                params=dict(n_directions=model.n_directions, gene_set=model.gene_set, psc=str(model.psc), qubits=model.qubits,
                                            scale=model.scale, n_neighbors=model.n_neighbors, tol=model.tol, backend=model.backend,
                                            leiden_seed=model.leiden_seed, donor_key=donor_key), resolution=model.resolution_)
    return model
