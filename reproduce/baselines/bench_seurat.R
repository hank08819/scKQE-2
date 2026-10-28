# Seurat v5 standard workflow on the shared subsample: LogNormalize, 2,000 vst HVGs, ScaleData, PCA 50 -> emb_seurat.csv
args <- commandArgs(trailingOnly = TRUE); d <- args[1]
suppressMessages({library(Seurat); library(Matrix)})
t0 <- Sys.time(); M <- readMM(file.path(d, "counts.mtx")); M <- as(M, "CsparseMatrix")
rownames(M) <- make.unique(readLines(file.path(d, "genes.tsv"))); colnames(M) <- readLines(file.path(d, "cells.tsv"))
s <- CreateSeuratObject(M); s <- NormalizeData(s, verbose = FALSE); s <- FindVariableFeatures(s, nfeatures = 2000, verbose = FALSE)
s <- ScaleData(s, verbose = FALSE); s <- RunPCA(s, npcs = 50, verbose = FALSE, seed.use = 0)
write.table(Embeddings(s, "pca"), file.path(d, "emb_seurat.csv"), sep = ",", col.names = FALSE, row.names = FALSE)
cat("seurat secs", as.numeric(difftime(Sys.time(), t0, units = "secs")), "\n")
