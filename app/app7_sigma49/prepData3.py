import sys
import os

sys.path.append('../..')
import prepLib

in_strFastaFilename = '{!s}/data/protein/sigma_49/Sigma_49_sequence.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/sigma_49/Sigma_49.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = '../app4_sigma49/protRefs' # for reuse, maybe should copy it here
out_strOutputBaseDir = './sparseData3'

#prepLib.breakFasta(in_strFastaFilename, in_strProtRefsDir, 0)
listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)
#listProtRefFileName = ['IPI00025499.1.txt', 'gi.txt']

# load peptide probabilities
listPepProb = prepLib.loadPepProbsFromCsv(in_strPeptideFilename, " ", 1, 3)
listPepProb = prepLib.consolidatePepProbs(listPepProb)

# match peptides with proteins
metaInfo = prepLib.fuRunAllProt_CleavageSites(listProtRefFileName, in_strProtRefsDir, out_strOutputBaseDir, listPepProb)

strMetaInfoFilename = '{!s}/metaInfo.csv'.format(out_strOutputBaseDir)
prepLib.fuSaveMetaInfo_CleavageSites(strMetaInfoFilename, metaInfo)
prepLib.fuSavePepProbsTargetFromList('{!s}/target.csv'.format(out_strOutputBaseDir), listPepProb) 
