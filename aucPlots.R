#!/usr/local/bin/Rscript

library(ggplot2)
library(grid)
library(gridExtra)
library(flux)

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
	df$auc <- auc(df$fpr, df$tpr)

	df$method <- paste(format(df$method, width=8, trim=FALSE), format(round(df$auc, digits=2), nsmall=2, justify="left"))
#	df$method <- paste(format(df$method, width=8, trim=FALSE), format(df$auc, digits=2, justify="left"))

	return(df)
}

loadPRFileInfo <- function(info)
{
	strFilename <- info[2]
	df <- read.table(strFilename, header=FALSE, col.names=c('precision', 'recall'))
	df <- df[order(df$precision),]
	df$method <- info[1]
	df$lineTheme <- info[3]

	df$auc <- auc(df$recall, df$precision )
	df$method <- paste(format(df$method, width=8, trim=FALSE), format(round(df$auc, digits=2), nsmall=2, justify="left"))

	return(df)
}

rocPlot <- function(roc_files)
{
	dfAll <- data.frame(fpr=numeric(), tpr=numeric(), method=factor(), lineTheme=numeric())
	for (curr in roc_files)
	{
    curr_df =loadROCFileInfo(curr)
		dfAll <- rbind(dfAll, curr_df)
	}

	plot_out <- ggplot(dfAll, aes(x=fpr, y=tpr, colour=method, size = lineTheme))+
                geom_line( )+
								scale_size_manual(values=c(d_S,d_L))+
								scale_colour_brewer(palette = "Set1")+
								guides(size = FALSE, colour = guide_legend(title="Method              AUC", override.aes=list(size = c(d_L, d_S, d_S, d_S, d_S))  ) )+
								theme_bw()+ 
								theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank())+
								theme(legend.text=element_text(family = "mono", size=12))+
								theme(legend.position=c(0.8, 0.15))+
								theme(legend.title=element_text(face = "bold"))
#                labs(title = strTitle)

	return(plot_out)
}

prPlot <- function(pr_files)
{
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
								guides(size = FALSE, colour = guide_legend(title="Method              AUC", override.aes=list(size = c(d_L, d_S, d_S, d_S, d_S))  ) )+
								theme_bw()+ theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank())+
								theme(legend.text=element_text(family = "mono"))+
								theme(legend.text=element_text(family = "mono", size=12))+
								theme(legend.position=c(0.2, 0.15))+
								theme(legend.title=element_text(face = "bold"))
#								theme(legend.position="none")
#                labs(title = strTitle)

	return(plot_out)
}


lineTheme_default=1
lineTheme_myMethod=2

# ********* Sigma49
roc_files <- list(c('DPep', 'app/app7_sigma49/sparseData3/protInfo_expr_17.csv.roc', lineTheme_myMethod),
             c('PLasso', '../ProteinLasso/real_data/Sigma_49_result.csv.roc', lineTheme_default),
             c('PLinear','../ProteinLP/real_data/Sigma_49_result.csv.roc', lineTheme_default),
             c('Fido','../fido/real_data/Sigma_49_result.csv.roc', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/Sigma_49_result.csv.roc', lineTheme_default))
plot_roc <- rocPlot(roc_files)

pr_files <- list(c('DPep', 'app/app7_sigma49/sparseData3/protInfo_expr_17.csv.pr', lineTheme_myMethod),
             c('PLasso', '../ProteinLasso/real_data/Sigma_49_result.csv.pr', lineTheme_default),
             c('PLinear','../ProteinLP/real_data/Sigma_49_result.csv.pr', lineTheme_default),
             c('Fido','../fido/real_data/Sigma_49_result.csv.pr', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/Sigma_49_result.csv.pr', lineTheme_default))
plot_pr <- prPlot(pr_files)
plot_sigma49 <- grid.arrange(plot_roc, plot_pr, widths= c(1, 1))
ggsave("sigma49.pdf", plot_sigma49, width=13.4, height = 6.89, path=dstDir)


# ********* 18mix
roc_files <- list(c('DPep', 'app/app8_18mix/sparseData3/protInfo_expr_17.csv.roc', lineTheme_myMethod),
             c('PLasso', '../ProteinLasso/real_data/18mix_result.csv.roc', lineTheme_default),
             c('PLinear','../ProteinLP/real_data/18mix_result.csv.roc', lineTheme_default),
             c('Fido','../fido/real_data/18mix_result.csv.roc', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/18mix_result.csv.roc', lineTheme_default))
plot_roc <- rocPlot(roc_files)

pr_files <- list(c('DPep', 'app/app8_18mix/sparseData3/protInfo_expr_17.csv.pr', lineTheme_myMethod),
             c('PLasso', '../ProteinLasso/real_data/18mix_result.csv.pr', lineTheme_default),
             c('PLinear','../ProteinLP/real_data/18mix_result.csv.pr', lineTheme_default),
             c('Fido','../fido/real_data/18mix_result.csv.pr', lineTheme_default),
             c('MSBayes','../MSBayesPro/real_data/18mix_result.csv.pr', lineTheme_default))
plot_pr <- prPlot(pr_files)
plot_18mix<- grid.arrange(plot_roc, plot_pr, widths= c(1, 1))
ggsave("18mix.pdf", plot_18mix, width=13.4, height = 6.89, path=dstDir)


# ********* Yeast
roc_files = list(c("DPep", "app/app9_yeast_pl/sparseData3/protInfo_expr_17.csv.roc", lineTheme_myMethod),
             c("PLasso", "../ProteinLasso/real_data/Yeast_result.csv.roc", lineTheme_default),
             c("PLinear","../ProteinLP/real_data/Yeast_result.csv.roc", lineTheme_default),
             c("Fido","../fido/real_data/Yeast_result.csv.roc", lineTheme_default),
             c("MSBayes","../MSBayesPro/real_data/Yeast_result.csv.roc", lineTheme_default))
plot_roc <- rocPlot(roc_files)
pr_files = list(c("DPep", "app/app9_yeast_pl/sparseData3/protInfo_expr_17.csv.pr", lineTheme_myMethod),
             c("PLasso", "../ProteinLasso/real_data/Yeast_result.csv.pr", lineTheme_default),
             c("PLinear","../ProteinLP/real_data/Yeast_result.csv.pr", lineTheme_default),
             c("Fido","../fido/real_data/Yeast_result.csv.pr", lineTheme_default),
             c("MSBayes","../MSBayesPro/real_data/Yeast_result.csv.pr", lineTheme_default))
plot_pr <- prPlot(pr_files)
plot_Yeast<- grid.arrange(plot_roc, plot_pr, widths= c(1, 1))
ggsave("Yeast.pdf", plot_Yeast, width=13.4, height = 6.89, path=dstDir)

