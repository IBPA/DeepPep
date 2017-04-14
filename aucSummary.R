#!/usr/local/bin/Rscript

library(ggplot2)
library(grid)
library(gridExtra)
library(flux)
library(plyr)

dstDir = "./data_comb/"

dfSummary = data.frame(method=character(), auc=numeric(), dataset=character(), best=numeric(), diff=numeric(),  stringsAsFactors=FALSE )
dfSummary[nrow(dfSummary) + 1,] <- c("DeepPep", 0.890, "yeast", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("Fido", 0.486, "yeast", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("PL", 0.897, "yeast", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("PLP", 0.856, "yeast", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("MSBayes", 0.769, "yeast", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("DeepPep", 0.977, "18mix", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("Fido", 0.962, "18mix", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("PL", 0.950, "18mix", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("PLP", 0.781, "18mix", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("MSBayes", 0.742, "18mix", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("DeepPep", 0.932, "sigma49", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("Fido", 0.931, "sigma49", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("PL", 0.906, "sigma49", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("PLP", 0.885, "sigma49", 0.0, 0.0)
dfSummary[nrow(dfSummary) + 1,] <- c("MSBayes", 0.897, "sigma49", 0.0, 0.0)

# processing dfSummary
dfSummary$method = as.factor(dfSummary$method)
dfSummary$dataset = as.factor(dfSummary$dataset)
dfSummary$auc = as.numeric(dfSummary$auc)
dfSummary$best = 0.0
dfSummary$diff = 0.0

print(dfSummary)

for (dataset in levels(dfSummary$dataset)) {
	df_sub <- dfSummary[dfSummary$dataset == dataset,]
	df_sub$best <- max(df_sub$auc)
	df_sub$diff <- (df_sub$best - df_sub$auc)
	dfSummary[dfSummary$dataset == dataset,] <- df_sub
}

dfAggr <- ddply(dfSummary, c("method"), summarise, diffMean = mean(diff), diffSd = sd(diff))
print(dfAggr)

plot_summary <- ggplot(dfAggr, aes(x=method, y=diffMean, ymin=diffMean-diffSd/2, ymax=diffMean+diffSd/2))+
								geom_bar(position="dodge", stat="identity")+
								geom_errorbar(position="dodge", width=0.1)+
								geom_text(aes(label=as.character(round(diffMean,3))), vjust=-0.7, hjust=-0.2 )+
								coord_flip()+
								ylab("Average difference of AUC from best performing method in each dataset")+
								xlab("")+
								theme_bw()+
								theme(text=element_text(size=16))+
								theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank())+
								theme(legend.text=element_text(family = "mono", size=12))

ggsave("roc_summary_DeepPep.pdf", plot_summary, width=13.4, height = 6.89, path=dstDir)


