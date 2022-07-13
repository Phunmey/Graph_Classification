setwd("C:/Code/result_26_5_22")
library(tidyverse)
library(dplyr)
library(fs)
library(ggplot2)
library(Hmisc)
library(cowplot)
library(tools)


file_paths <- fs::dir_ls("C:/Code/result_26_5_22")
file_paths


file_contents <- list()
seq_along(file_paths)

for (i in seq_along(file_paths)) {
  file_contents[[i]] <- read.csv(file = file_paths[[i]])
  df <- file_contents[[i]]
  filename <- file_path_sans_ext(basename(file_paths[[i]]))
  
  names(df) <- 
    c("dataset",
      "filtr_time",
      "training_time",
      "accuracy",
      "auc",
      "thresh",
      "step_size",
      "flat_conf_mat") #give column names
  df <-
    subset(df, select = -c(flat_conf_mat)) #remove last column
  
  sort_result <- df %>%
    dplyr::group_by(dataset, step_size) %>%
    dplyr::summarise(
      avgacc = mean(accuracy) * 100,
      stdacc = sd(accuracy) * 100,
      avgauc = mean(auc) * 100,
      stdauc = sd(auc) * 100,
      avgfiltrtime = mean(filtr_time),
      avgtraintime = mean(training_time),
      .groups = "keep"
    )  #sort results based on the dataset and the step_size
  
  sum_by_max <-
    sort_result %>% dplyr::group_by(dataset) %>% 
    dplyr::slice(which.max(avgacc)) %>%
    dplyr::mutate_if(is.numeric, round, 2)
    #sort again using maximum average auc of each group


  latex(
    sum_by_max,
    caption = filename,
    file = paste("result_26_5", ".tex", sep = ""),
    append = TRUE,
    label = filename,
    rowname = NULL,
    center = "centering"
  ) #write to latex file

 #  plt <- ggplot(sort_result, aes(
 #    x = step_size,
 #    y = avgauc,
 #    #group_by = dataset,
 #    color = dataset
 #  )) + geom_line()
 # 
 #  plt
 # 
 # ggsave(paste("graph/",toString(filename),".png",sep=""))

  
}
