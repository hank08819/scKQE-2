# Reproducing the results of the paper

All atlases are public on CZ CELLxGENE Discover; `atlases.csv` lists the dataset version ids of Extended Data Table 8.
Install the package with `pip install "scKQE2[gpu]"`. Complete atlases need a machine with a large memory; in our runs the exact
decomposition of the 11.4-million-nucleus mouse embryo atlas used up to about 390 GB on a 502-GB server.

## 1. Data

```bash
python download.py fetal --out data          # one atlas;  python download.py all --out data  for every atlas
```

## 2. KQE-2 on a complete atlas (Table 1, Extended Data Table 2)

```bash
python run_kqe2.py data/fetal.h5ad --label cell_type --out results/fetal --save-scores
python run_kqe2.py data/developing_cortex.h5ad --label cell_type --k 8 --out results/developing_cortex
```

The script applies the cell filter of the paper (unannotated cells and types with fewer than 10 cells removed), the gene-set
rule, the exact decomposition, the entanglement-free encoding and the clustering of Table 1 (Precision Shrinkage Clustering on
whole-transcriptome atlases, Leiden at k on 5,000-HVG atlases; Leiden seed 0), and prints the AMI. The developing cortex is
scored at k = 8 (`atlases.csv`, column k_scoring). Running the script twice on one machine gives identical clusters.

## 3. Quantum-state maps (Fig. 3)

```bash
python run_kqe2.py data/siletti_nonneuronal.h5ad --out results/siletti --save-scores
python state_map.py results/siletti_scores.npy results/siletti_clusters.csv.gz --out results/siletti
```

Prints the number of nuclei certified by Theorem 4, the share of the 15 nearest states in the own cluster and type, and writes
the map. The same commands on `mouse_embryo_complete` give Fig. 3e,f.

## 4. Rare embryonic types (Results, Rare embryonic types)

```bash
python run_kqe2.py data/mouse_embryo_complete.h5ad --out results/mouse_embryo_complete
python rare_markers.py data/mouse_embryo_complete.h5ad results/mouse_embryo_complete_clusters.csv.gz
```

## 5. Baselines (Table 1)

`baselines/` holds the scripts used for the other methods, all scored by the same protocol (`score_emb.py`: exact 15-nearest-
neighbour graph, UMAP fuzzy weights, Leiden bisected to k). Table 1 reports Leiden seed 0.

| Method | Script | Settings |
|---|---|---|
| scanpy | `bench_python.py <dir> scanpy` | normalization to 10^4, log1p, 2,000 HVGs, scaling clipped at 10, 50 PCs |
| scVI 1.4.2 | `bench_python.py <dir> scvi` | 2,000 HVGs, raw counts, 30 latent dimensions, seed 0, early stopping |
| Seurat 5.3.0 | `Rscript bench_seurat.R <dir>` | NormalizeData, 2,000 variable features, ScaleData, 50 PCs |
| scGPT | `bench_scgpt.py <dir>` | whole-human checkpoint (`SCGPT_MODEL` environment variable), raw counts |

`bench_subsample.py` writes the input folder (`counts.mtx`, `cells.tsv`, `labels.tsv`, `sub.h5ad`) for a complete atlas or for the
fixed 150,000-cell subsample of the paper (`KQE_SUB=150000`).

## Hardware

`sckqe2.hardware.block_circuit` gives the circuit of one block; the paper ran 380 developing-cortex cells on the IBM processor
ibm_marrakesh (Heron r2) with qiskit-ibm-runtime EstimatorV2, 1,024 shots, readout-error mitigation at resilience level 1.
