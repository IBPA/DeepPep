import pandas as pd
from ggplot import *

def loadAUCFileInfo(info):
    strName=info[0]
    strFilename=info[1]
    df=pd.read_csv(strFilename, header=None, sep=' ', names=['tpr', 'fpr'])
    return strName, df

roc_files = [['PLasso', '../ProteinLasso/real_data/Sigma_49_result.csv.roc'],
              ['PLinear','../ProteinLP/real_data/Sigma_49_result.csv.roc']]

#for curr in roc_files:
curr_strName1, curr_df1 =loadAUCFileInfo(roc_files[0])
curr_strName2, curr_df2 =loadAUCFileInfo(roc_files[1])
plot_out = ggplot(curr_df1, aes(x='tpr', y='fpr')) +\
            geom_line(color="blue") +\
            geom_line(data=curr_df2, color="red") +\
            geom_abline(linetype='dashed')

plot_out.save("roc.pdf")


