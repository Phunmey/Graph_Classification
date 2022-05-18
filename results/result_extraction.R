library(tidyverse)
library(dplyr)
library(fs)
library(ggplot2)
library(Hmisc)
library(cowplot)
library(tools)
setwd("C:/Code/New folder")

file_paths <- fs::dir_ls("C:/Code/New folder/New_Results")
file_paths


file_contents <- list()
seq_along(file_paths)

for (i in seq_along(file_paths)) {
  file_contents[[i]] <- read.csv(file = file_paths[[i]])
  df <- file_contents[[i]]
  filename <- file_path_sans_ext(basename(file_paths[[i]]))
  
  names(df) <- 
    c("dataset",
      "kerneltime",
      "traintime",
      "accuracy",
      "auc",
      "iter",
      "flat_conf_mat") #give column names
  df <-
    subset(df, select = -c(flat_conf_mat)) #remove last 2 columns
  
  sort_result <- df %>%
    dplyr::group_by(dataset, iter) %>%
    dplyr::summarise(
      avgacc = mean(accuracy) * 100,
      stdacc = sd(accuracy) * 100,
      avgauc = mean(auc) * 100,
      stdauc = sd(auc) * 100,
      avgkerneltime = mean(kerneltime),
      avgtraintime = mean(traintime),
      .groups = "keep"
    )  #sort results based on the dataset and the number of iterations
  
  sum_by_max <-
    sort_result %>% dplyr::group_by(dataset) %>% 
    dplyr::slice(which.max(avgauc)) %>%
    dplyr::mutate_if(is.numeric, round, 2)
    #sort again using maximum average auc of each group
  

  latex(
    sum_by_max,
    caption = filename,
    file = paste("finalresult", ".tex", sep = ""),
    append = TRUE,
    label = filename,
    rowname = NULL,
    center = "centering"
  ) #write to latex file
  
  plt <- ggplot(sort_result, aes( 
    x = iter,
    y = avgauc,
    group_by = dataset,
    color = dataset
  )) + geom_line() 
  
  plt
  
  ggsave(paste("graph/",toString(filename),".png",sep=""))
  
  
}
