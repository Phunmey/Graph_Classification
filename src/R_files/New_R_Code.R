library(tidyverse)
require(plyr)
require(ggplot2)
library("dplyr")
library(fs)
#library(kableExtra)
#library(Hmisc)
library(tools)
library(cowplot)

setwd("C:/Code/results")
#data <- read.csv("C:/Code/results/New_Results/Eigenvalueresults_csv.csv", header = T)

file_paths <- fs::dir_ls("C:/Code/New folder/Sort_results")
file_paths


file_contents <- list()
seq_along(file_paths)

for (i in seq_along(file_paths)) {
  file_contents[[i]] <- read.csv(file = file_paths[[i]])
  df <- file_contents[[i]]
  filename <- file_path_sans_ext(basename(file_paths[[i]]))
  
  d2 <-
    ddply(df,
      .(C1, C6, C7),
      summarise,
      acc_m = 100 * mean(C4),
      acc_std = 100 * sd(C4),
      auc_m = 100 * mean(C5),
      auc_std = 100 * sd(C5),
      n = length(C5)
    )
  plt <- ggplot(d2, aes(
    x = C6,
    y = auc_m,
    group_by = C1,
    color = C1
  )) + geom_line() + scale_x_continuous(breaks = c(2, 3, 4))+
    labs(title = toString(filename))
  
  plt
  
  #ggsave(paste("graph/",toString(filename),".png",sep=""))
}
