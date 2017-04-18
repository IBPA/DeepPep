# output: sparseData2: target.csv, metaInfo.csv, *.txt
# input-verify-against: sigma_49_reference.csv
# input-protein-reference:  Sigma_49_sequence.fasta  
# input-peptides: ~/data/protein/sigma_49/Sigma_49.txt
#a) 

import sys
import os
sys.path.append('../..')
import prepLib
in_strFastaFilename = '{!s}/data/protein/18mix/18mix_db_plus_contaminants_20081209.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/18mix/18_mixtures_peptide_identification.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = './protRefs'
out_strOutputBaseDir = './sparseData2'

protsDic = prepLib.loadUniqProtsDicFromCsv(in_strPeptideFilename, "\t", 1)
prepLib.breakFasta(in_strFastaFilename, in_strProtRefsDir, 1, protsDic)
listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)

# load peptide probabilities
listPepProb = prepLib.loadPepProbsFromCsv(in_strPeptideFilename, "\t", 0, 2)
listPepProb = prepLib.consolidatePepProbs(listPepProb)



# match peptides with proteins
prepLib.fuRunAllProt(listProtRefFileName, in_strProtRefsDir, out_strOutputBaseDir, listPepProb)


strMetaInfoFilename = '{!s}/metaInfo.csv'.format(out_strOutputBaseDir)
prepLib.fuSaveMetaInfo(out_strOutputBaseDir, strMetaInfoFilename, in_strProtRefsDir)
prepLib.fuSavePepProbsTargetFromList('{!s}/target.csv'.format(out_strOutputBaseDir), listPepProb) 
