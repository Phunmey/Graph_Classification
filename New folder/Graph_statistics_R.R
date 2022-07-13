library(tidyverse)
library(dplyr)
library(fs)
library(ggplot2)
library(Hmisc)
library(cowplot)
library(tools)
setwd("C:/Code/New folder")

#file_paths <- fs::dir_ls("C:/Code")
#file_paths

graph_file <- read.csv(file = New_Resultsgraph_statistics)

names(graph_file) <- c("dataset", "avg.density", "avg.distance", "avg.clustering coeff.")

latex(
  graph_file,
  caption = graph_statistics,
  file = paste("graph stat", ".tex", sep = ""),
  append = TRUE,
  rowname = NULL,
  center = "centering"
)