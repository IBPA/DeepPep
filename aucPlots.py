import pandas as pd
from ggplot import *

def loadROCFileInfo(info):
    strFilename=info[1]
    df=pd.read_csv(strFilename, header=None, sep=' ', names=['tpr', 'fpr'])
    df['method']=info[0] 
    df['size']=info[2]

    return df

def rocPlot(roc_files, strFileFigure, strTitle):
    dfAll=pd.DataFrame(columns=['tpr', 'fpr', 'method'])
    for curr in roc_files:
        curr_df =loadROCFileInfo(curr)
        dfAll=dfAll.append(curr_df, ignore_index=True)

    plot_out = ggplot(dfAll, aes(x='tpr', y='fpr', colour='method')) +\
                geom_line( aes(size = 'size')) +\
                labs(title = strTitle) +\
                theme_bw() + theme(panel_border = None)

    plot_out.save(strFileFigure)

def loadPRFileInfo(info):
    strFilename=info[1]
    df=pd.read_csv(strFilename, header=None, sep=' ', names=['precision', 'recall'])
    df['method']=info[0] 
    df['size']=info[2]

    return df

def prPlot(roc_files, strFileFigure, strTitle):
    dfAll=pd.DataFrame(columns=['precision', 'recall', 'method'])
    for curr in roc_files:
        curr_df =loadPRFileInfo(curr)
        dfAll=dfAll.append(curr_df, ignore_index=True)

    plot_out = ggplot(dfAll, aes(x='recall', y='precision', colour='method')) +\
                geom_line( aes(size = 'size')) +\
                labs(title = strTitle) +\
                theme_bw() + theme(panel_border = None)

    plot_out.save(strFileFigure)



thickness_default=1
thickness_thick=2.5

# ********* Sigma49
roc_files = [['DPep', 'app/app4_sigma49/sparseData2/protInfo_expr_5.csv.roc', thickness_thick],
             ['PLasso', '../ProteinLasso/real_data/Sigma_49_result.csv.roc', thickness_default],
             ['PLinear','../ProteinLP/real_data/Sigma_49_result.csv.roc', thickness_default],
             ['Fido','../fido/real_data/Sigma_49_result.csv.roc', thickness_default],
             ['MSBayes','../MSBayesPro/real_data/Sigma_49_result.csv.roc', thickness_default]]
rocPlot(roc_files, './data_comb/Sigma49_roc.pdf', "Sigma49")
pr_files = [['DPep', 'app/app4_sigma49/sparseData2/protInfo_expr_5.csv.pr', thickness_thick],
             ['PLasso', '../ProteinLasso/real_data/Sigma_49_result.csv.pr', thickness_default],
             ['PLinear','../ProteinLP/real_data/Sigma_49_result.csv.pr', thickness_default],
             ['Fido','../fido/real_data/Sigma_49_result.csv.pr', thickness_default],
             ['MSBayes','../MSBayesPro/real_data/Sigma_49_result.csv.pr', thickness_default]]
prPlot(pr_files, './data_comb/Sigma49_pr.pdf', "Sigma49")


# ********* 18mix
roc_files = [['DPep', 'app/app5_18mix/sparseData2/protInfo_expr_3.csv.roc', thickness_thick],
             ['PLasso', '../ProteinLasso/real_data/18mix_result.csv.roc', thickness_default],
             ['PLinear','../ProteinLP/real_data/18mix_result.csv.roc', thickness_default],
             ['Fido','../fido/real_data/18mix_result.csv.roc', thickness_default],
             ['MSBayes','../MSBayesPro/real_data/18mix_result.csv.roc', thickness_default]]
rocPlot(roc_files, './data_comb/18mix_roc.pdf', "18mix")
pr_files = [['DPep', 'app/app5_18mix/sparseData2/protInfo_expr_3.csv.pr', thickness_thick],
             ['PLasso', '../ProteinLasso/real_data/18mix_result.csv.pr', thickness_default],
             ['PLinear','../ProteinLP/real_data/18mix_result.csv.pr', thickness_default],
             ['Fido','../fido/real_data/18mix_result.csv.pr', thickness_default],
             ['MSBayes','../MSBayesPro/real_data/18mix_result.csv.pr', thickness_default]]
prPlot(pr_files, './data_comb/18mix_pr.pdf', "18mix")


# ********* Yeast
roc_files = [['DPep', 'app/app6_yeast_pl/sparseData2/protInfo_expr_3.csv.roc', thickness_thick],
             ['PLasso', '../ProteinLasso/real_data/Yeast_result.csv.roc', thickness_default],
             ['PLinear','../ProteinLP/real_data/Yeast_result.csv.roc', thickness_default],
             ['Fido','../fido/real_data/Yeast_result.csv.roc', thickness_default],
             ['MSBayes','../MSBayesPro/real_data/Yeast_result.csv.roc', thickness_default]]
rocPlot(roc_files, './data_comb/Yeast_roc.pdf', "Yeast")
pr_files = [['DPep', 'app/app6_yeast_pl/sparseData2/protInfo_expr_3.csv.pr', thickness_thick],
             ['PLasso', '../ProteinLasso/real_data/Yeast_result.csv.pr', thickness_default],
             ['PLinear','../ProteinLP/real_data/Yeast_result.csv.pr', thickness_default],
             ['Fido','../fido/real_data/Yeast_result.csv.pr', thickness_default],
             ['MSBayes','../MSBayesPro/real_data/Yeast_result.csv.pr', thickness_default]]
prPlot(pr_files, './data_comb/Yeast_pr.pdf', "Yeast")
