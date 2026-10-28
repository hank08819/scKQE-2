"""scKQE-2: reproducible quantum clustering of atlas-scale single-cell data on the whole transcriptome.

scKQE-2 turns each cell into a quantum state that holds all of its genes and groups cells whose states are close.
No step is trained and no step draws a random number: the same counts give the same clusters on every run.
"""
from .preprocess import to_csr32, idf_matrix_inplace, hvg_index
from .decompose import exact_scores
from .encode import encode, bloch_vectors, fidelity, lipschitz
from .cluster import knn, knn_graph, leiden, leiden_at_k
from .align import quantum_align
from .rule import choose_gene_set, signal_entropy, data_entropy
from .api import KQE2, kqe2
from .psc import leiden_psc, merge_to_k
from .qstate import certified_knn, state_knn, state_map, state_distance, neighbor_purity, bloch_chunked

__version__ = "2.0.0"
__author__ = "Henry Han"
__email__ = "hank08819@gmail.com"
__all__ = ["KQE2", "kqe2", "leiden_psc", "merge_to_k", "certified_knn", "state_knn", "state_map", "state_distance", "neighbor_purity", "bloch_chunked", "to_csr32", "idf_matrix_inplace", "hvg_index", "exact_scores", "encode", "bloch_vectors", "fidelity",
           "lipschitz", "knn", "knn_graph", "leiden", "leiden_at_k", "quantum_align", "choose_gene_set", "signal_entropy",
           "data_entropy"]
