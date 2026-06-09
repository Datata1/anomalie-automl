# Konvertiert die vier TEP-.RData-Dateien nach Parquet (TICKET-10).
#
# Voraussetzung: die vier .RData-Dateien liegen unter data/ (Download via Kaggle/Harvard
# Dataverse, siehe docs/methoden/01_datensatz_tep.md §6). Beim Laden entstehen die Variablen
# fault_free_training, fault_free_testing, faulty_training, faulty_testing.
#
# Ausführen:  Rscript export_rdata.R

library(arrow)

load("data/TEP_FaultFree_Training.RData")
load("data/TEP_FaultFree_Testing.RData")
load("data/TEP_Faulty_Training.RData")
load("data/TEP_Faulty_Testing.RData")

write_parquet(fault_free_training, "data/TEP_FaultFree_Training.parquet")
write_parquet(fault_free_testing,  "data/TEP_FaultFree_Testing.parquet")
write_parquet(faulty_training,     "data/TEP_Faulty_Training.parquet")
write_parquet(faulty_testing,      "data/TEP_Faulty_Testing.parquet")

cat("Fertig: 4 Parquet-Dateien unter data/ geschrieben.\n")
