# =============================================================================
# Task 2: CpG-Gene Association Annotation
# Output: cpg_gene_edges.tsv (evidence edge table) and cpg_gene_summary.tsv (summary table)
# Genome version: hg19 (consistent throughout pipeline)
# =============================================================================

library(data.table)
library(GenomicRanges)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(org.Hs.eg.db)

# ---- Configuration ----
genome_version <- "hg19"
genehancer_file <- "genehancer/genehancer_Interactions_DE_hg19.bed"
bios_eqtm_file <- "cis_eQTMsFDR0.05-CpGLevel.txt"
tss_window_size <- 10000  # +/-10kb

# ---- 1. Read EWAS results ----
ewas <- fread("EWASresults.tsv")
cpg_unique <- unique(ewas[, .(CpG, Chr, Pos)])
cpg_unique <- cpg_unique[!is.na(Chr) & !is.na(Pos)]

cpg_gr <- GRanges(
  seqnames = paste0("chr", cpg_unique$Chr),
  ranges = IRanges(start = cpg_unique$Pos, width = 1),
  CpG = cpg_unique$CpG
)

edges_list <- list()

# ---- 2. Method 1: TSS window-based gene association ----
txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
genes_txdb <- genes(txdb)
tss_gr <- resize(genes_txdb, width = 1, fix = "start")
tss_window <- resize(tss_gr, width = 2 * tss_window_size + 1, fix = "center")

overlaps <- findOverlaps(cpg_gr, tss_window)

cpg_positions <- start(cpg_gr)[queryHits(overlaps)]
tss_positions <- start(tss_gr)[subjectHits(overlaps)]
distances <- abs(cpg_positions - tss_positions)

gene_ids <- names(genes_txdb)[subjectHits(overlaps)]
gene_symbols <- mapIds(org.Hs.eg.db, keys = gene_ids,
                       column = "SYMBOL", keytype = "ENTREZID",
                       multiVals = "first")

tss_edges <- data.table(
  CpG = cpg_gr$CpG[queryHits(overlaps)],
  Gene = gene_symbols,
  Gene_ID = gene_ids,
  evidence_type = "TSS_window",
  source = paste0("TxDb.hg19.+/-", tss_window_size/1000, "kb"),
  score = NA_real_,
  distance = distances,
  pval = NA_real_,
  fdr = NA_real_
)
tss_edges <- tss_edges[!is.na(Gene)]
edges_list[["TSS"]] <- tss_edges

# ---- 3. Method 2: Read GREAT results ----
great_result_file <- "great_region_gene_edges.tsv"

great_raw <- fread(great_result_file)

great_edges <- data.table(
    CpG = great_raw$CpG,
    Gene = great_raw$Gene,
    Gene_ID = NA_character_,
    evidence_type = "GREAT",
    source = "rGREAT_hg19",
    score = NA_real_,
    distance = abs(great_raw$dist_to_TSS),
    pval = NA_real_,
    fdr = NA_real_
  )
great_edges <- great_edges[CpG %in% cpg_unique$CpG]
edges_list[["GREAT"]] <- great_edges

# ---- 4. Method 3: GeneHancer enhancer-gene association ----
gh_raw <- fread(genehancer_file, header = FALSE)

colnames(gh_raw) <- c("chr", "start", "end", "gene_enhancer", "score")
gh_raw[, c("gene_name", "enhancer_id") := tstrsplit(gene_enhancer, "/", fixed = TRUE)]

gh_gr <- GRanges(
  seqnames = gh_raw$chr,
  ranges = IRanges(start = gh_raw$start + 1, end = gh_raw$end),
  gene = gh_raw$gene_name,
  enhancer_id = gh_raw$enhancer_id,
  score = gh_raw$score
)

overlaps <- findOverlaps(cpg_gr, gh_gr)

genehancer_edges <- data.table(
  CpG = cpg_gr$CpG[queryHits(overlaps)],
  Gene = gh_gr$gene[subjectHits(overlaps)],
  Gene_ID = NA_character_,
  evidence_type = "GeneHancer_DoubleElite",
  source = "UCSC_GeneHancer_hg19",
  score = gh_gr$score[subjectHits(overlaps)],
  distance = NA_integer_,
  pval = NA_real_,
  fdr = NA_real_
)
edges_list[["GeneHancer"]] <- genehancer_edges

# ---- 5. Method 4: BIOS eQTM data ----
# eQTM = expression Quantitative Trait Methylation
# Column mapping (legacy naming): SNPName=CpG, ProbeName=Ensembl_ID, HGNCName=Gene_Symbol
bios <- fread(bios_eqtm_file)

bios_sig <- if ("FDR" %in% names(bios)) bios[FDR < 0.05] else bios

eqtm_edges <- data.table(
  CpG = bios_sig$SNPName,
  Gene = bios_sig$HGNCName,
  Gene_ID = bios_sig$ProbeName,
  evidence_type = "eQTM",
  source = "BIOS_consortium",
  score = abs(bios_sig$OverallZScore),
  distance = NA_integer_,
  pval = bios_sig$PValue,
  fdr = bios_sig$FDR
)
eqtm_edges <- eqtm_edges[!is.na(Gene) & Gene != ""]
edges_list[["eQTM"]] <- eqtm_edges

# ---- 6. Merge all edges and create evidence table ----
all_edges <- rbindlist(edges_list, fill = TRUE)

# Priority: eQTM > GeneHancer > GREAT > TSS_window
priority_map <- c(eQTM = 1, GeneHancer_DoubleElite = 2, GREAT = 3, TSS_window = 4)
all_edges[, priority := priority_map[evidence_type]]

edges_output_file <- "cpg_gene_edges.tsv"
fwrite(all_edges[, .(CpG, Gene, Gene_ID, evidence_type, source, score, distance, pval, fdr, priority)],
       edges_output_file, sep = "\t")

# ---- Save separate edge tables by evidence type ----

save_edge_table <- function(key, cols, new_names, filename) {
  if (key %in% names(edges_list)) {
    output <- edges_list[[key]][, ..cols]
    setnames(output, cols, new_names)
    fwrite(output, filename, sep = "\t")
  }
}

save_edge_table("TSS", c("CpG", "Gene", "Gene_ID", "distance"),
                c("CpG", "TSS_Gene", "Entrez_ID", "Distance_to_TSS"),
                "cpg_gene_edges_TSS_window.tsv")

save_edge_table("GREAT", c("CpG", "Gene", "distance"),
                c("CpG", "GREAT_Gene", "Distance_to_TSS"),
                "cpg_gene_edges_GREAT.tsv")

save_edge_table("GeneHancer", c("CpG", "Gene", "score"),
                c("CpG", "Enhancer_Target_Gene", "GeneHancer_Score"),
                "cpg_gene_edges_GeneHancer.tsv")

save_edge_table("eQTM", c("CpG", "Gene", "Gene_ID", "score", "pval", "fdr"),
                c("CpG", "eQTM_Gene", "Ensembl_ID", "Z_Score", "P_Value", "FDR"),
                "cpg_gene_edges_eQTM.tsv")

# ---- 7. Create summary table with Best_Gene selection ----

# Priority: eQTM > GeneHancer > GREAT > TSS (nearest)
# Within same priority: lowest FDR > highest score > shortest distance
select_best_gene <- function(edges_dt) {
  if (nrow(edges_dt) == 0) return(NA_character_)

  setorder(edges_dt, priority)
  best_edges <- edges_dt[priority == edges_dt$priority[1]]

  if (nrow(best_edges) == 1) return(best_edges$Gene[1])

  if (any(!is.na(best_edges$fdr))) {
    return(best_edges[order(fdr, na.last = TRUE)]$Gene[1])
  }
  if (any(!is.na(best_edges$score))) {
    return(best_edges[order(-score, na.last = TRUE)]$Gene[1])
  }
  if (any(!is.na(best_edges$distance))) {
    return(best_edges[order(distance, na.last = TRUE)]$Gene[1])
  }
  best_edges$Gene[1]
}

best_genes <- all_edges[, .(
  Best_Gene = select_best_gene(.SD),
  Best_Evidence = evidence_type[which.min(priority)],
  N_Evidence_Types = uniqueN(evidence_type),
  N_Total_Edges = .N,
  All_Genes = paste(unique(Gene), collapse = ";"),
  Evidence_Types = paste(unique(evidence_type), collapse = ";")
), by = CpG]

cpg_summary <- merge(cpg_unique, best_genes, by = "CpG", all.x = TRUE)

# Add per-evidence-type summaries
eqtm_summary <- all_edges[evidence_type == "eQTM", .(
  eQTM_Genes = paste(unique(Gene), collapse = ";"),
  eQTM_Best_FDR = min(fdr, na.rm = TRUE)
), by = CpG]
eqtm_summary[is.infinite(eQTM_Best_FDR), eQTM_Best_FDR := NA]

gh_summary <- all_edges[evidence_type == "GeneHancer_DoubleElite", .(
  GeneHancer_Genes = paste(unique(Gene), collapse = ";"),
  GeneHancer_Best_Score = max(score, na.rm = TRUE)
), by = CpG]
gh_summary[is.infinite(GeneHancer_Best_Score), GeneHancer_Best_Score := NA]

great_summary <- all_edges[evidence_type == "GREAT", .(
  GREAT_Genes = paste(unique(Gene), collapse = ";")
), by = CpG]

tss_summary <- all_edges[evidence_type == "TSS_window", .(
  TSS_Genes = paste(unique(Gene), collapse = ";"),
  TSS_Nearest_Dist = min(distance, na.rm = TRUE)
), by = CpG]
tss_summary[is.infinite(TSS_Nearest_Dist), TSS_Nearest_Dist := NA]

cpg_summary <- Reduce(function(x, y) merge(x, y, by = "CpG", all.x = TRUE),
                      list(cpg_summary, eqtm_summary, gh_summary, great_summary, tss_summary))

# ---- 8. Save results ----
cpg_summary_filtered <- cpg_summary[!is.na(Best_Gene) & Best_Gene != ""]

summary_output_file <- "cpg_gene_summary.tsv"
fwrite(cpg_summary_filtered, summary_output_file, sep = "\t")

# ---- 9. Statistics ----
cat("\n=== CpG-Gene Association Statistics ===\n")
cat(sprintf("Genome version: %s\n", genome_version))
cat(sprintf("Total CpGs (filtered): %d\n", nrow(cpg_summary_filtered)))

cat("\n--- Evidence Type Coverage ---\n")
n_total <- nrow(cpg_summary_filtered)

count_coverage <- function(col_name, label) {
  if (col_name %in% names(cpg_summary_filtered)) {
    n <- sum(!is.na(cpg_summary_filtered[[col_name]]) & cpg_summary_filtered[[col_name]] != "")
    cat(sprintf("%s: %d (%.1f%%)\n", label, n, 100 * n / n_total))
  }
}

count_coverage("eQTM_Genes", "eQTM associations")
count_coverage("GeneHancer_Genes", "GeneHancer associations")
count_coverage("GREAT_Genes", "GREAT associations")
count_coverage("TSS_Genes", sprintf("TSS window (+/-%dkb)", tss_window_size/1000))

n_best <- sum(!is.na(cpg_summary_filtered$Best_Gene) & cpg_summary_filtered$Best_Gene != "")
cat(sprintf("\nWith Best_Gene: %d (%.1f%%)\n", n_best, 100 * n_best / n_total))

if ("Best_Evidence" %in% names(cpg_summary_filtered)) {
  cat("\n--- Best_Gene Source Distribution ---\n")
  print(table(cpg_summary_filtered$Best_Evidence, useNA = "ifany"))
}

cat("\n--- Edge Table Statistics ---\n")
cat(sprintf("Total edges: %d\n", nrow(all_edges)))
cat("Edges by evidence type:\n")
print(all_edges[, .N, by = evidence_type][order(-N)])