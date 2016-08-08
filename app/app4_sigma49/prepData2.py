# output: sparseData2: target.csv, metaInfo.csv, *.txt
# input-verify-against: sigma_49_reference.csv
# input-protein-reference:  Sigma_49_sequence.fasta  
# input-peptides: ~/data/protein/sigma_49/Sigma_49.txt
#a) 

import sys
import os
sys.path.append('../..')
import prepLib

in_strFastaFilename = '{!s}/data/protein/sigma_49/Sigma_49_sequence.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/sigma_49/Sigma_49.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = './protRefs'
out_strOutputBaseDir = './sparseData2'

#prepLib.breakFasta(in_strFastaFilename, in_strProtRefsDir)
listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)

# load peptide probabilities
listPepProb = prepLib.loadPepProbsFromCsv(in_strPeptideFilename, " ", 1, 3)
listPepProb = prepLib.consolidatePepProbs(listPepProb)



# match peptides with proteins
#prepLib.fuRunAllProt(listProtRefFileName[0:-1], in_strProtRefsDir, out_strOutputBaseDir, listPepProb)


strMetaInfoFilename = '{!s}/metaInfo.csv'.format(out_strOutputBaseDir)
prepLib.fuSaveMetaInfo(out_strOutputBaseDir, strMetaInfoFilename, in_strProtRefsDir)
prepLib.fuSavePepProbsTargetFromList('{!s}/target.csv'.format(out_strOutputBaseDir), listPepProb) 
