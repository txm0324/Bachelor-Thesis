if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("edgeR")
library(edgeR)

TMM_normalization <- function (count_data, return_instance, coeficient) {
  cd_numeric <- data.matrix(count_data)
  n_genes = ncol(count_data)
  n_samples = nrow(count_data)
  
  cd_numeric_t = t(cd_numeric)
  count_cell_lines = matrix(data=cd_numeric_t, nrow=n_genes, ncol=n_samples)

  assign('count_data', count_cell_lines)

  D <- DGEList(counts=count_data)
  Dnorm <- calcNormFactors(D)

  coef_parameters = Dnorm

  rellibsize <- colSums (D$counts)/exp(mean(log(colSums(D$counts)))) 
  nf <- Dnorm$samples [,3]*rellibsize
  TMM_normalized_data = round (sweep(D$counts, 2, nf, "/"))
  TMM_normalized_data_T = t(TMM_normalized_data)

  return(TMM_normalized_data_T)
}

# Which df you use in read_data (in data_settings)
rnaseq_UMG <- read.csv("./final_new_GDSC.csv")

# Run Code and save result
rnaseq_UMG_tmm <- TMM_normalization(rnaseq_UMG, FALSE, NA)
rnaseq_UMG_tmm_log <- log(rnaseq_UMG_tmm + 1) # Avoid -Inf because of log(0 + 1) = 0.
write.csv(rnaseq_UMG_tmm_log, file="./TMM_log_new_GDSC_normalized_rnaseq.csv")
