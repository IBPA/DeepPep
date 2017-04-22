# Prerequisite: directories for "in_strProtRefsDir" and "sparseData2", should not contain any ".txt" file
# Output: under sparseData2 directory: target.csv, metaInfo.csv, *.txt

import sys
import os
sys.path.append('../..')
import prepLib
in_strFastaFilename = '{!s}/data/protein/18mix/18mix_db_plus_contaminants_20081209.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/18mix/18_mixtures_peptide_identification.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = './protRefs'
out_strOutputBaseDir = './sparseData2'

protDic, pepDic = prepLib.loadProtPeptideDic(in_strPeptideFilename)
prepLib.breakFasta(in_strFastaFilename, in_strProtRefsDir, protDic)
listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)

# match peptides with proteins
prepLib.fuRunAllProt(listProtRefFileName, in_strProtRefsDir, out_strOutputBaseDir, protDic)

strMetaInfoFilename = '{!s}/metaInfo.csv'.format(out_strOutputBaseDir)
prepLib.fuSaveMetaInfo(out_strOutputBaseDir, strMetaInfoFilename, in_strProtRefsDir)
pepProbsList = sorted(list(pepDic.values()),key=lambda x: x[0])
pepProbsList = [pepProbsList[i][1:3] for i in range(0,len(pepProbsList))]
prepLib.fuSavePepProbsTargetFromList('{!s}/target.csv'.format(out_strOutputBaseDir), pepProbsList)
