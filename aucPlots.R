#!/usr/local/bin/Rscript

library(ggplot2)
library(grid)
library(gridExtra)
library(flux)
library(plyr)

d_L <- 1.5
d_S <- 0.5
dstDir = "./data_comb/"

loadROCFileInfo <- function(info)
{
	strFilename <- info[2]
	df <- read.table(strFilename, header=FALSE, col.names=c('fpr', 'tpr'))
	df <- df[order(df$tpr),]
	df$method <- info[1]
	df$lineTheme <- info[3]

	dAUC = auc(df$fpr, df$tpr)
	cat(sprintf("%s, %f\n", info[1], dAUC))
	df$auc <- dAUC

	df$method <- paste(format(df$method, width=8, trim=FALSE), format(round(df$auc, digits=2), nsmall=2, justify="left"))
#	df$method <- paste(format(df$method, width=8, trim=FALSE), format(df$auc, digits=2, justify="left"))

	res = list("df"= df, "dAUC"= dAUC)

	return(res)
}

loadPRFileInfo <- function(info)
{
	strFilename <- info[2]
	df <- read.table(strFilename, header=FALSE, col.names=c('precision', 'recall'))
	df <- df[order(df$precision),]
	df$method <- info[1]
	df$lineTheme <- info[3]

	dAUC = auc(df$recall, df$precision )
	cat(sprintf("%s, %f\n", info[1], dAUC))

	df$auc <- dAUC
	df$method <- paste(format(df$method, width=8, trim=FALSE), format(round(df$auc, digits=2), nsmall=2, justify="left"))

	return(df)
}

rocPlot <- function(roc_files)
{
	print("ROC:")

	dfAll <- data.frame(fpr=numeric(), tpr=numeric(), method=factor(), lineTheme=numeric())
	dfSummary = data.frame(method=character(), auc=numeric(), stringsAsFactors=FALSE )
	for (curr in roc_files)
	{
		res <- loadROCFileInfo(curr)
		curr_df = res$df
		dfAll <- rbind(dfAll, curr_df)

		curr_dAUC <- res$dAUC
		dfSummary[nrow(dfSummary) + 1,] <-  c(curr[1], curr_dAUC)
	}

	plot_out <- ggplot(dfAll, aes(x=fpr, y=tpr, colour=method, size = lineTheme))+
                geom_line( )+
								scale_size_manual(values=c(d_S,d_L))+
								scale_colour_brewer(palette = "Set1")+
								guides(size = FALSE, colour = guide_legend(title="Method              AUC", override.aes=list(size = c(d_S, d_S, d_S, d_S, d_L))  ) )+
								theme_bw()+ 
								theme(text=element_text(size=16))+
								theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank())+
								theme(legend.text=element_text(family = "mono"))+#, size=12))+
								theme(legend.position=c(0.8, 0.15))+
								theme(legend.title=element_text(face = "bold", size=12))
#                labs(title = strTitle)

	res <- list("plot" = plot_out, "summary" = dfSummary)
	return(res)
}

prPlot <- function(pr_files)
{
	print("PR:")

	dfAll <- data.frame(precision=numeric(), recall=numeric(), method=factor(), lineTheme=numeric())
	for (curr in pr_files)
	{
    curr_df =loadPRFileInfo(curr)
		dfAll <- rbind(dfAll, curr_df)
	}

	plot_out <- ggplot(dfAll, aes(y=precision, x=recall, colour=method, size = lineTheme))+
                geom_line( )+
								scale_size_manual(values=c(d_S,d_L))+
								scale_colour_brewer(palette = "Set1")+
								guides(size = FALSE, colour = guide_legend(title="Method              AUC", override.aes=list(size = c(d_S, d_S, d_S, d_S, d_L))  ) )+
								theme_bw()+ 
								theme(text=element_text(size=16))+
								theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank())+
								theme(legend.text=element_text(family = "mono"))+#, size=12))+
								theme(legend.position=c(0.2, 0.15))+
								theme(legend.title=element_text(face = "bold", size=12))
#								theme(legend.position="none")
#                labs(title = strTitle)

	return(plot_out)
}


dfSummary = data.frame(method=character(), auc=numeric(), dataset=character(), stringsAsFactors=FALSE )
appendSummary <- function(dfMain, dfNew, strDataset)
{
	dfNew$dataset = strDataset
	dfMain <- rbind(dfMain, dfNew)

	return(dfMain)
}


lineTheme_default=1
lineTheme_myMethod=2

# ********* Sigma49
print("** Sigma49")
roc_files <- list(c("WidePep", 'app/app4_sigma49/sparseData2/protInfo_expr_23.csv.roc', lineTheme_myMethod),
             c('ProteinLasso', '../ProteinLasso/real_data/Sigma_49_result.csv.roc', lineTheme_default),
             c('ProteinLP','../ProteinLP/real_data/Sigma_49_result.csv.roc', lineTheme_default),
             c('Fido','../fido/real_data/Sigma_49_result.csv.roc', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/Sigma_49_result.csv.roc', lineTheme_default))
res <- rocPlot(roc_files)
plot_roc <- res$plot
dfSummary <- appendSummary(dfSummary, res$summary, "sigma49")

pr_files <- list(c("WidePep", 'app/app4_sigma49/sparseData2/protInfo_expr_23.csv.pr', lineTheme_myMethod),
             c('ProteinLasso', '../ProteinLasso/real_data/Sigma_49_result.csv.pr', lineTheme_default),
             c('ProteinLP','../ProteinLP/real_data/Sigma_49_result.csv.pr', lineTheme_default),
             c('Fido','../fido/real_data/Sigma_49_result.csv.pr', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/Sigma_49_result.csv.pr', lineTheme_default))
plot_pr <- prPlot(pr_files)
plot_sigma49 <- grid.arrange(plot_roc, plot_pr, widths= c(1, 1))
ggsave("sigma49.pdf", plot_sigma49, width=13.4, height = 6.89, path=dstDir)


# ********* 18mix
print("** 18mix")
roc_files <- list(c("WidePep", 'app/app5_18mix/sparseData2/protInfo_expr_23.csv.roc', lineTheme_myMethod),
             c('ProteinLasso', '../ProteinLasso/real_data/18mix_result.csv.roc', lineTheme_default),
             c('ProteinLP','../ProteinLP/real_data/18mix_result.csv.roc', lineTheme_default),
             c('Fido','../fido/real_data/18mix_result.csv.roc', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/18mix_result.csv.roc', lineTheme_default))
res <- rocPlot(roc_files)
plot_roc <- res$plot
dfSummary <- appendSummary(dfSummary, res$summary, "18mix")

pr_files <- list(c("WidePep", 'app/app5_18mix/sparseData2/protInfo_expr_23.csv.pr', lineTheme_myMethod),
             c('ProteinLasso', '../ProteinLasso/real_data/18mix_result.csv.pr', lineTheme_default),
             c('ProteinLP','../ProteinLP/real_data/18mix_result.csv.pr', lineTheme_default),
             c('Fido','../fido/real_data/18mix_result.csv.pr', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/18mix_result.csv.pr', lineTheme_default))
plot_pr <- prPlot(pr_files)
plot_18mix<- grid.arrange(plot_roc, plot_pr, widths= c(1, 1))
ggsave("18mix.pdf", plot_18mix, width=13.4, height = 6.89, path=dstDir)


# ********* Yeast
print("** Yeast")
roc_files = list(c("WidePep", "app/app6_yeast_pl/sparseData2/protInfo_expr_23.csv.roc", lineTheme_myMethod),
             c("ProteinLasso", "../ProteinLasso/real_data/Yeast_result.csv.roc", lineTheme_default),
             c("ProteinLP","../ProteinLP/real_data/Yeast_result.csv.roc", lineTheme_default),
             c("Fido","../fido/real_data/Yeast_result.csv.roc", lineTheme_default),
             c("MSBayes","../MSBayesPro/real_data/Yeast_result.csv.roc", lineTheme_default))
res <- rocPlot(roc_files)
plot_roc <- res$plot
dfSummary <- appendSummary(dfSummary, res$summary, "yeast")

pr_files = list(c("WidePep", "app/app6_yeast_pl/sparseData2/protInfo_expr_23.csv.pr", lineTheme_myMethod),
             c("ProteinLasso", "../ProteinLasso/real_data/Yeast_result.csv.pr", lineTheme_default),
             c("ProteinLP","../ProteinLP/real_data/Yeast_result.csv.pr", lineTheme_default),
             c("Fido","../fido/real_data/Yeast_result.csv.pr", lineTheme_default),
             c("MSBayes","../MSBayesPro/real_data/Yeast_result.csv.pr", lineTheme_default))
plot_pr <- prPlot(pr_files)
plot_Yeast<- grid.arrange(plot_roc, plot_pr, widths= c(1, 1))
ggsave("Yeast.pdf", plot_Yeast, width=13.4, height = 6.89, path=dstDir)


# processing dfSummary
dfSummary$method = as.factor(dfSummary$method)
dfSummary$dataset = as.factor(dfSummary$dataset)
dfSummary$auc = as.numeric(dfSummary$auc)
dfSummary$best = 0.0
dfSummary$diff = 0.0

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

ggsave("roc_summary.pdf", plot_summary, width=13.4, height = 6.89, path=dstDir)


