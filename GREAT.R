#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(GenomicRanges)
  library(IRanges)
  library(rGREAT)
})

infile  <- "EWASresults.tsv"
outfile <- "great_region_gene_edges.tsv"

dt <- fread(infile, sep = "\t", data.table = TRUE)

required_cols <- c("CpG", "Chr", "Pos")
missing_cols <- setdiff(required_cols, names(dt))
if (length(missing_cols) > 0) {
  stop("Missing columns in EWASresults.tsv: ", paste(missing_cols, collapse = ", "))
}

dt <- dt[!is.na(Chr) & !is.na(Pos)]
dt <- unique(dt, by = "CpG")

chr <- as.character(dt$Chr)
chr <- ifelse(grepl("^chr", chr, ignore.case = TRUE), chr, paste0("chr", chr))
chr <- sub("^CHR", "chr", chr)

gr <- GRanges(
  seqnames = chr,
  ranges = IRanges(start = as.integer(dt$Pos), width = 1)
)
names(gr) <- dt$CpG

obj <- great(gr, gene_sets = "GO:BP", tss_source = "txdb:hg19", cores = 1)
asso <- getRegionGeneAssociations(obj, use_symbols = TRUE)

mcols_names <- names(mcols(asso))

genes_col <- if ("annotated_genes" %in% mcols_names) {
  "annotated_genes"
} else if ("gene" %in% mcols_names) {
  "gene"
} else {
  stop("Cannot find gene list column in associations.")
}

dist_col <- if ("dist_to_TSS" %in% mcols_names) {
  "dist_to_TSS"
} else if ("distTSS" %in% mcols_names) {
  "distTSS"
} else {
  stop("Cannot find distance column in associations.")
}

genes_list <- mcols(asso)[[genes_col]]
dist_list <- mcols(asso)[[dist_col]]

lens <- elementNROWS(genes_list)
idx <- rep.int(seq_along(genes_list), lens)

edge_dt <- data.table(
  CpG = names(asso)[idx],
  Gene = unlist(genes_list, use.names = FALSE),
  dist_to_TSS = unlist(dist_list, use.names = FALSE)
)

fwrite(edge_dt, outfile, sep = "\t")
cat("Wrote:", outfile, "\n")
