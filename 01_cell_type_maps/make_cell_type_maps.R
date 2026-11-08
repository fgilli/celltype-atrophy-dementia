# ============================================================
# make_cell_type_maps.R
#
# Builds the eight cell-type maps on the fsaverage6 surface.
#
# Inputs
#   1) Vertex-wise interpolated mRNA expression maps of Ecker et al.
#      (lh.rh.mRNA.fsavg6.fwhm5.rda; list with $lh and $rh, each a
#      genes x vertices matrix).
#   2) Cell-type marker gene sets of Seidlitz et al. (2020),
#      Supplementary Data 5 (xlsx with columns Type, Paper, Cluster,
#      Class and one column per marker gene).
#
# Method
#   For every cell type, the expression map of each marker gene is
#   z-scored across vertices and the cell-type map is the vertex-wise
#   mean over all marker genes that are present in the expression data.
#
# Outputs
#   <out_prefix>_lh.feather, <out_prefix>_rh.feather
#       one row per vertex, column "vertex" (0-based) plus one column
#       per cell type
#   <out_prefix>_metadata.json
#       number of vertices/genes and marker overlap per cell type
#
# The fsaverage6 maps were subsequently reduced to the fsaverage5 (ico5)
# vertex set and stored as fsavg5_{hemi}_{cell}.npy for the downstream
# Python pipeline.
#
# Usage
#   Rscript make_cell_type_maps.R [mRNA_rda] [seidlitz_xlsx] [out_prefix]
# ============================================================

suppressPackageStartupMessages({
  library(arrow)
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
mRNA_rda      <- if (length(args) >= 1) args[1] else "lh.rh.mRNA.fsavg6.fwhm5.rda"
seidlitz_xlsx <- if (length(args) >= 2) args[2] else "Seidlitz2020_SuppData5_celltype_genes.xlsx"
out_prefix    <- if (length(args) >= 3) args[3] else "fsavg6_cellclass_proxies"
normalize     <- "z"   # "z" (z-score each marker gene across vertices) or "none"

stop_if_missing <- function(path, label = "file") {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) {
    stop(sprintf("Missing %s: %s", label, path), call. = FALSE)
  }
}

make_proxy_map <- function(mat, genes, normalize = c("z", "none")) {
  normalize <- match.arg(normalize)
  genes_found <- intersect(genes, rownames(mat))
  if (length(genes_found) == 0) stop("No marker genes found in matrix rownames().")

  sub <- mat[genes_found, , drop = FALSE]

  if (normalize == "z") {
    mu  <- rowMeans(sub, na.rm = TRUE)
    sdv <- apply(sub, 1, sd, na.rm = TRUE)
    sdv[is.na(sdv) | sdv == 0] <- 1
    sub <- (sub - mu) / sdv
  }

  proxy <- colMeans(sub, na.rm = TRUE)
  attr(proxy, "n_markers_found") <- length(genes_found)
  attr(proxy, "markers_found")   <- genes_found
  attr(proxy, "markers_missing") <- setdiff(genes, rownames(mat))
  proxy
}

read_seidlitz_markers <- function(xlsx_path) {
  sheets <- excel_sheets(xlsx_path)
  if (length(sheets) < 1) stop("No sheets found in Seidlitz xlsx.", call. = FALSE)

  header <- c("Type", "Paper", "Cluster", "Class")
  df <- NULL
  for (s in sheets) {
    tmp <- suppressWarnings(read_xlsx(xlsx_path, sheet = s))
    if (all(header %in% colnames(tmp))) {
      df <- tmp
      break
    }
  }
  if (is.null(df)) {
    stop("Could not find expected columns (Type, Paper, Cluster, Class) in Seidlitz xlsx.", call. = FALSE)
  }

  gene_cols <- setdiff(colnames(df), header)

  long <- df %>%
    pivot_longer(cols = all_of(gene_cols), names_to = "gene_col", values_to = "gene") %>%
    mutate(gene = str_trim(as.character(gene))) %>%
    filter(!is.na(gene), gene != "")

  markers_by_class <- long %>%
    group_by(Class) %>%
    summarise(genes = list(sort(unique(gene))), .groups = "drop")

  setNames(markers_by_class$genes, markers_by_class$Class)
}

make_df <- function(proxy_list) {
  data.frame(
    vertex = seq_along(proxy_list[[1]]) - 1L,
    as.data.frame(proxy_list, check.names = FALSE)
  )
}

normalize <- match.arg(normalize, c("z", "none"))
stop_if_missing(mRNA_rda, "mRNA_rda")
stop_if_missing(seidlitz_xlsx, "seidlitz_xlsx")

cat("== Building fsaverage6 cell-type maps ==\n")
cat("mRNA_rda      :", mRNA_rda, "\n")
cat("seidlitz_xlsx :", seidlitz_xlsx, "\n")
cat("out_prefix    :", out_prefix, "\n")
cat("normalize     :", normalize, "\n\n")

ecker_env <- new.env(parent = emptyenv())
load(mRNA_rda, envir = ecker_env)
if (!exists("lh.rh.mRNA.fsavg6.fwhm5", envir = ecker_env)) {
  stop("Object 'lh.rh.mRNA.fsavg6.fwhm5' not found in mRNA RDA.", call. = FALSE)
}
mRNA <- get("lh.rh.mRNA.fsavg6.fwhm5", envir = ecker_env)
if (!is.list(mRNA) || !all(c("lh", "rh") %in% names(mRNA))) {
  stop("Expected 'lh.rh.mRNA.fsavg6.fwhm5' to be a list with $lh and $rh.", call. = FALSE)
}
lh <- mRNA$lh
rh <- mRNA$rh
if (!is.matrix(lh) || !is.matrix(rh)) {
  stop("Expected mRNA$lh and mRNA$rh to be matrices (genes x vertices).", call. = FALSE)
}
cat("mRNA dims (genes x vertices): lh", paste(dim(lh), collapse = " x "),
    "| rh", paste(dim(rh), collapse = " x "), "\n\n")

markers_all <- read_seidlitz_markers(seidlitz_xlsx)
cat("Marker classes found:", paste(sort(names(markers_all)), collapse = ", "), "\n\n")

required <- c("Astro", "Endo", "Micro", "OPC", "Oligo", "Neuro-Ex", "Neuro-In", "Per")
missing_req <- setdiff(required, names(markers_all))
if (length(missing_req) > 0) {
  stop("Missing required marker class(es) in Seidlitz table: ",
       paste(missing_req, collapse = ", "), call. = FALSE)
}

markers8 <- list(
  astrocytes       = markers_all[["Astro"]],
  endothelial      = markers_all[["Endo"]],
  microglia        = markers_all[["Micro"]],
  opc              = markers_all[["OPC"]],
  oligodendrocytes = markers_all[["Oligo"]],
  excitatory       = markers_all[["Neuro-Ex"]],
  inhibitory       = markers_all[["Neuro-In"]],
  pericytes        = markers_all[["Per"]]
)

overlap_counts <- sapply(markers8, \(g) sum(g %in% rownames(lh)))
cat("Marker genes present in the expression data:\n")
print(overlap_counts)
cat("\n")

proxy_lh <- lapply(markers8, \(g) make_proxy_map(lh, g, normalize = normalize))
proxy_rh <- lapply(markers8, \(g) make_proxy_map(rh, g, normalize = normalize))

df_lh <- make_df(proxy_lh)
df_rh <- make_df(proxy_rh)

out_lh   <- paste0(out_prefix, "_lh.feather")
out_rh   <- paste0(out_prefix, "_rh.feather")
out_meta <- paste0(out_prefix, "_metadata.json")

write_feather(df_lh, out_lh)
write_feather(df_rh, out_rh)

meta <- list(
  space = "fsaverage6",
  n_vertices = ncol(lh),
  n_genes = nrow(lh),
  normalize = normalize,
  marker_source = "Seidlitz et al. 2020 Supplementary Data 5 (cell-type gene sets; Class column)",
  mRNA_source = "Ecker et al. fsaverage6 interpolated mRNA expression maps (lh.rh.mRNA.fsavg6.fwhm5.rda)",
  columns = names(df_lh),
  marker_overlap_counts = overlap_counts,
  created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
writeLines(toJSON(meta, auto_unbox = TRUE, pretty = TRUE), out_meta)

cat("Done. Outputs:\n  ", out_lh, "\n  ", out_rh, "\n  ", out_meta, "\n")
