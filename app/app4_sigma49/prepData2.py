# output: sparseData2: target.csv, metaInfo.csv, *.txt
# input-verify-against: sigma_49_reference.csv
# input-protein-reference:  Sigma_49_sequence.fasta  
# input-peptides: ~/data/protein/sigma_49/Sigma_49.txt
#a) 

import sys
import os
sys.path.append('../..')
import prepLib
from pprint import pprint

in_strFastaFilename = '{!s}/data/protein/sigma_49/Sigma_49_sequence.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/sigma_49/Sigma_49.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = './protRefs'
out_strOutputBaseDir = './sparseData2'

protsDic = prepLib.loadUniqProtsDicFromCsv(in_strPeptideFilename, "\t", 2)
prepLib.breakFasta(in_strFastaFilename, in_strProtRefsDir, 0, protsDic)
listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)

# load peptide probabilities
listPepProb = prepLib.loadPepProbsFromCsv(in_strPeptideFilename, " ", 1, 3)
listPepProb = prepLib.consolidatePepProbs(listPepProb)

#pprint(listPepProb)
#listPepProb = [['AEISMLEGAVLDIR', 1.0]]

# match peptides with proteins
prepLib.fuRunAllProt(listProtRefFileName, in_strProtRefsDir, out_strOutputBaseDir, listPepProb)


strMetaInfoFilename = '{!s}/metaInfo.csv'.format(out_strOutputBaseDir)
prepLib.fuSaveMetaInfo(out_strOutputBaseDir, strMetaInfoFilename, in_strProtRefsDir)
prepLib.fuSavePepProbsTargetFromList('{!s}/target.csv'.format(out_strOutputBaseDir), listPepProb) 
