import sys
import os

sys.path.append('../..')
import prepLib4
import prepLib

in_strFastaFilename = '{!s}/data/protein/sigma_49/Sigma_49_sequence.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/sigma_49/Sigma_49.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = '../app4_sigma49/protRefs' # for reuse, maybe should copy it here
in_strDetectabilitiesFilename = '{!s}/mygithub/ProteinLasso/real_data/Sigma_49_peptide_detectability.txt'.format(os.environ.get('HOME'))
out_strOutputBaseDir = './sparseData4'

#prepLib.breakFasta(in_strFastaFilename, in_strProtRefsDir, 0)
listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)
#listProtRefFileName = ['P06396.txt']

# load peptide probabilities
listPepProb = prepLib.loadPepProbsFromCsv(in_strPeptideFilename, " ", 1, 3)
listPepProb = prepLib.consolidatePepProbs(listPepProb)
listPepProb = prepLib4.appendDetectabilitiesFromCsv(listPepProb, in_strDetectabilitiesFilename, "\t", 0, 2)

# match peptides with proteins
metaInfo = prepLib4.fuRunAllProt_CleavageSites(listProtRefFileName, in_strProtRefsDir, out_strOutputBaseDir, listPepProb)
print(metaInfo)
strMetaInfoFilename = '{!s}/metaInfo.csv'.format(out_strOutputBaseDir)
prepLib.fuSaveMetaInfo_CleavageSites(strMetaInfoFilename, metaInfo)
prepLib4.fuSavePepProbsTargetFromList('{!s}/target.csv'.format(out_strOutputBaseDir), listPepProb) 
