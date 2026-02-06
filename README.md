## CpG Biological Function Annotation

### Workflow

```
EWASresults.tsv (Input)
        │
        ▼
┌───────────────────────────────────────┐
│  1. Extract unique CpG sites          │
│     (CpG ID, Chr, Pos, Type)          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  2. Convert to GRanges object         │
│     (Genomic coordinate system)       │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  3. Annotate using annotatr package   │
│     ├─ CpG Island context             │
│     │   (Island/Shore/Shelf/OpenSea)  │
│     ├─ Gene structure features        │
│     │   (Promoter/Exon/Intron/IGR)    │
│     └─ FANTOM5 Enhancers              │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  4. ChromHMM Chromatin State          │
│     (Roadmap Epigenomics E062-Blood)  │
│     15-state model annotation         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  5. Merge all annotations             │
└───────────────────────────────────────┘
        │
        ▼
cpg_biological_annotations.tsv (Output)
```

### Data Sources

| Annotation Type | Source | Description |
|----------------|--------|-------------|
| CpG Island Features | EWAS input (`Type` column) | Island, Shore, Shelf, Open Sea |
| Gene Structure | `annotatr` (hg19_basicgenes) | Promoter, Exon, Intron, Intergenic |
| Enhancers | `annotatr` (hg19_enhancers_fantom) | FANTOM5 enhancer regions |
| Chromatin State | Roadmap ChromHMM | 15-state chromatin segmentation (E062 = Blood) |

### ChromHMM 15-State Model

| State | Name | Biological Meaning |
|-------|------|-------------------|
| 1_TssA | Active TSS | Active transcription start sites |
| 2_TssAFlnk | Flanking Active TSS | Regions flanking active TSS |
| 3_TxFlnk | Transcr. at gene 5'/3' | Transcription at gene boundaries |
| 4_Tx | Strong Transcription | Actively transcribed regions |
| 5_TxWk | Weak Transcription | Weakly transcribed regions |
| 6_EnhG | Genic Enhancers | Enhancers within gene bodies |
| 7_Enh | Enhancers | Intergenic enhancers |
| 8_ZNF/Rpts | ZNF genes & Repeats | Zinc finger genes and repetitive elements |
| 9_Het | Heterochromatin | Condensed, inactive chromatin |
| 10_TssBiv | Bivalent/Poised TSS | Poised promoters (developmental genes) |
| 11_BivFlnk | Flanking Bivalent TSS/Enh | Flanking bivalent regions |
| 12_EnhBiv | Bivalent Enhancer | Poised enhancers |
| 13_ReprPC | Repressed PolyComb | Polycomb-repressed regions |
| 14_ReprPCWk | Weak Repressed PolyComb | Weakly repressed regions |
| 15_Quies | Quiescent/Low | Inactive/low signal regions |

### Output Columns

| Column | Description |
|--------|-------------|
| CpG | CpG probe ID (e.g., cg00000029) |
| Chr | Chromosome number |
| Pos | Genomic position (hg19) |
| CpG_Island_Feature | Relation to CpG island |
| Gene_Feature | Gene structure annotation |
| Overlapping_Gene | Gene symbol if overlapping |
| Is_Enhancer | TRUE/FALSE for FANTOM5 enhancer |
| ChromState | Roadmap 15-state chromatin state |

---

## CpG-Gene Linkage Annotation

### Workflow

```
EWASresults.tsv (Input)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    4 Evidence Sources                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Method 1:       │  │ Method 2:       │                  │
│  │ TSS Window      │  │ GREAT           │                  │
│  │ (±10kb)         │  │ (Regulatory     │                  │
│  │                 │  │  Domains)       │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│  ┌────────┴────────┐  ┌────────┴────────┐                  │
│  │ Method 3:       │  │ Method 4:       │                  │
│  │ GeneHancer      │  │ BIOS eQTM       │                  │
│  │ (Enhancer-Gene  │  │ (Expression     │                  │
│  │  Interactions)  │  │  correlation)   │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Merge all evidence sources           │
│  Apply priority ranking               │
│  Select Best_Gene per CpG             │
└───────────────────────────────────────┘
        │
        ├──► cpg_gene_edges.tsv (All evidence)
        ├──► cpg_gene_edges_TSS_window.tsv
        ├──► cpg_gene_edges_GREAT.tsv
        ├──► cpg_gene_edges_GeneHancer.tsv
        ├──► cpg_gene_edges_eQTM.tsv
        └──► cpg_gene_summary.tsv (Summary with Best_Gene)
```

### Method Details

#### Method 1: TSS Window (±10kb)
- **Approach**: Find genes whose TSS is within ±10kb of the CpG site
- **Data Source**: `TxDb.Hsapiens.UCSC.hg19.knownGene`
- **Logic**: Uses `resize()` to get TSS position, then expands to ±10kb window
- **Output**: Gene symbol, Entrez ID, distance to TSS

#### Method 2: GREAT (Genomic Regions Enrichment of Annotations Tool)
- **Approach**: Assigns regulatory domains to genes based on GREAT algorithm
- **Data Source**: Pre-computed results from `GREAT.R` → `great_region_gene_edges.tsv`
- **Logic**: Each CpG is assigned to genes based on their basal + extended regulatory domains
- **Output**: Gene symbol, distance to TSS

#### Method 3: GeneHancer (Enhancer-Gene Interactions)
- **Approach**: Links CpGs in enhancer regions to their target genes
- **Data Source**: `genehancer_Interactions_DE_hg19.bed` (DoubleElite interactions)
- **Logic**: Overlaps CpG positions with GeneHancer enhancer-gene interaction regions
- **Output**: Target gene, GeneHancer confidence score

#### Method 4: BIOS eQTM (Expression Quantitative Trait Methylation)
- **Approach**: Links CpGs to genes based on methylation-expression correlations
- **Data Source**: `cis_eQTMsFDR0.05-CpGLevel.txt` (BIOS Consortium)
- **Logic**: Uses statistically significant (FDR < 0.05) methylation-expression associations
- **Output**: Gene symbol, Ensembl ID, Z-score, P-value, FDR

### Best Gene Selection Priority

The pipeline selects the most reliable gene link for each CpG using this priority:

| Priority | Evidence Type | Rationale |
|----------|--------------|-----------|
| 1 (Highest) | eQTM | Direct functional evidence (methylation affects expression) |
| 2 | GeneHancer DoubleElite | High-confidence enhancer-promoter interactions |
| 3 | GREAT | Regulatory domain-based assignment |
| 4 (Lowest) | TSS Window | Simple proximity-based assignment |

**Tie-breaking rules** (within same priority):
1. Lowest FDR (if available)
2. Highest score (if available)
3. Shortest distance (if available)

### Output Files

#### `cpg_gene_edges.tsv` (Complete Edge Table)
All CpG-gene associations with full evidence metadata.

| Column | Description |
|--------|-------------|
| CpG | CpG probe ID |
| Gene | Gene symbol |
| Gene_ID | Entrez ID or Ensembl ID |
| evidence_type | Source method |
| source | Specific database/tool |
| score | Confidence score (if available) |
| distance | Distance to TSS (if available) |
| pval | P-value (for eQTM) |
| fdr | FDR (for eQTM) |
| priority | Evidence priority (1-4) |

#### `cpg_gene_summary.tsv` (Summary Table)
One row per CpG with aggregated information.

| Column | Description |
|--------|-------------|
| CpG, Chr, Pos | CpG identification |
| Best_Gene | Selected gene based on priority |
| Best_Evidence | Evidence type of Best_Gene |
| N_Evidence_Types | Number of different evidence sources |
| N_Total_Edges | Total number of gene links |
| All_Genes | All linked genes (semicolon-separated) |
| eQTM_Genes | Genes from eQTM evidence |
| GeneHancer_Genes | Genes from GeneHancer |
| GREAT_Genes | Genes from GREAT |
| TSS_Genes | Genes from TSS window |


```
result/
├── cpg_biological_annotations.tsv    # Task 1 output
├── cpg_gene_edges.tsv                # Task 2: all edges
├── cpg_gene_edges_TSS_window.tsv     # Task 2: TSS method
├── cpg_gene_edges_GREAT.tsv          # Task 2: GREAT method
├── cpg_gene_edges_GeneHancer.tsv     # Task 2: GeneHancer method
├── cpg_gene_edges_eQTM.tsv           # Task 2: eQTM method
├── cpg_gene_summary.tsv              # Task 2: summary table
└── great_region_gene_edges.tsv       # GREAT pre-computed results
```