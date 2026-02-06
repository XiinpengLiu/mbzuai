# =============================================================================
# Task 1: CpG Site Biological Annotation
# Adds chromatin state, enhancer, and gene feature annotations to CpG sites
# Uses hg19 genome version throughout (required for Roadmap ChromHMM compatibility)
# =============================================================================

library(data.table)
library(GenomicRanges)
library(annotatr)
library(rtracklayer)

# Configuration
genome_version <- "hg19"
chromhmm_bed_file <- "E062_15_coreMarks_mnemonics.bed.gz"  # E062 = blood tissue

# 1. Read EWAS results
ewas <- fread("EWASresults.tsv")

cpg_unique <- unique(ewas[, .(CpG, Chr, Pos, CpG_Island_Feature = Type)])
cpg_unique <- cpg_unique[!is.na(Chr) & !is.na(Pos)]

# 2. Convert to GRanges
cpg_gr <- GRanges(
  seqnames = paste0("chr", cpg_unique$Chr),
  ranges = IRanges(start = cpg_unique$Pos, width = 1),
  CpG = cpg_unique$CpG
)

# 3. Genomic annotation with annotatr
annot_types <- c(
  paste0(genome_version, "_cpgs"),
  paste0(genome_version, "_basicgenes"),
  paste0(genome_version, "_enhancers_fantom")
)

annots_gr <- build_annotations(genome = genome_version, annotations = annot_types)

cpg_annotated <- annotate_regions(
  regions = cpg_gr,
  annotations = annots_gr,
  ignore.strand = TRUE,
  quiet = FALSE
)

annot_df <- as.data.frame(cpg_annotated)

# 4. Roadmap chromatin state annotation
chromhmm_dt <- fread(chromhmm_bed_file, header = FALSE,
                     col.names = c("chr", "start", "end", "state"))

chromhmm_gr <- GRanges(
  seqnames = chromhmm_dt$chr,
  ranges = IRanges(start = chromhmm_dt$start + 1, end = chromhmm_dt$end),  # BED is 0-based
  state = chromhmm_dt$state
)

overlaps <- findOverlaps(cpg_gr, chromhmm_gr)

cpg_chromstate <- data.table(
  CpG = cpg_gr$CpG[queryHits(overlaps)],
  ChromState = chromhmm_gr$state[subjectHits(overlaps)]
)
cpg_chromstate <- cpg_chromstate[!duplicated(CpG)]

# 5. Integrate all annotations
annot_dt <- data.table(annot_df)

gene_feature <- annot_dt[grepl("genes", annot.type), .(
  Gene_Feature = paste(unique(gsub(".*_genes_", "", annot.type)), collapse = ";"),
  Overlapping_Gene = paste(unique(na.omit(annot.symbol)), collapse = ";")
), by = CpG]

enhancer <- annot_dt[grepl("enhancer", annot.type), .(Is_Enhancer = "TRUE"), by = CpG]

annot_summary <- Reduce(function(x, y) merge(x, y, by = "CpG", all = TRUE),
                        list(gene_feature, enhancer))

annot_summary <- merge(annot_summary, cpg_chromstate, by = "CpG", all.x = TRUE)

cpg_final <- merge(cpg_unique, annot_summary, by = "CpG", all.x = TRUE)
cpg_final[is.na(Is_Enhancer), Is_Enhancer := "FALSE"]

# 6. Save results
output_file <- "cpg_biological_annotations.tsv"
fwrite(cpg_final, output_file, sep = "\t")

# 7. Summary statistics
cat("\n=== Annotation Summary ===\n")
cat(sprintf("Genome version: %s\n", genome_version))
cat(sprintf("Total CpG sites: %d\n", nrow(cpg_final)))

print_stat <- function(col_name, label, is_boolean = FALSE) {
  if (col_name %in% names(cpg_final)) {
    if (is_boolean) {
      count <- sum(cpg_final[[col_name]] == "TRUE", na.rm = TRUE)
    } else {
      count <- sum(!is.na(cpg_final[[col_name]]))
    }
    cat(sprintf("%s: %d (%.1f%%)\n", label, count, 100 * count / nrow(cpg_final)))
  }
}

print_stat("CpG_Island_Feature", "With CpG island annotation")
print_stat("Gene_Feature", "With gene feature annotation")
print_stat("Is_Enhancer", "In FANTOM5 enhancer", is_boolean = TRUE)
print_stat("ChromState", "With chromatin state annotation")

if ("ChromState" %in% names(cpg_final)) {
  cat("\nChromatin state distribution:\n")
  print(table(cpg_final$ChromState, useNA = "ifany"))
}