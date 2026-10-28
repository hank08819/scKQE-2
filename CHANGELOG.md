# Changelog

## 2.0.0
- Default pipeline is the configuration of the paper: the gene-set rule chooses the whole transcriptome or 5,000 HVGs
  (`gene_set="auto"`), and Precision Shrinkage Clustering is applied on whole-transcriptome atlases (`psc="auto"`).
- New `sckqe2.psc`: Precision Shrinkage Clustering (`leiden_psc`, `merge_to_k`).
- New `sckqe2.qstate`: exact quantum-state distance, nearest states certified by Theorem 4 (`certified_knn`, `state_knn`),
  maps built from exact fidelity (`state_map`), neighbour purity.
- New `sckqe2.hardware`: Qiskit circuit of one block, the observables read by KQE-2 and their exact values.
- New `reproduce/`: download of the atlases, KQE-2 on complete atlases, quantum-state maps, rare-type markers, baseline scripts.
- Command line: `--psc auto|on|off`; the gene-set rule runs inside `KQE2.fit_predict`.
- Tests: 12 (added PSC, certified nearest states, Theorem 4 bound, Qiskit circuit).

## 1.0.0
- First release: IDF weighting, exact decomposition, entanglement-free encoding (statevector and closed form), Leiden at k,
  quantum alignment, gene-set rule.
