import sys
import os

sys.path.append('../..')
import prepLib

in_strFastaFilename = '{!s}/data/protein/18mix/18mix_db_plus_contaminants_20081209.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/18mix/18_mixtures_peptide_identification.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = '../app5_18mix/protRefs' # for reuse, maybe should copy it here
out_strOutputBaseDir = './sparseData3'

#prepLib.breakFasta(in_strFastaFilename, in_strProtRefsDir, 0)
listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)
#listProtRefFileName = ['Q29443.txt']

# load peptide probabilities
listPepProb = prepLib.loadPepProbsFromCsv(in_strPeptideFilename, "\t", 0, 2)
listPepProb = prepLib.consolidatePepProbs(listPepProb)

# match peptides with proteins
metaInfo = prepLib.fuRunAllProt_CleavageSites(listProtRefFileName, in_strProtRefsDir, out_strOutputBaseDir, listPepProb)

strMetaInfoFilename = '{!s}/metaInfo.csv'.format(out_strOutputBaseDir)
prepLib.fuSaveMetaInfo_CleavageSites(strMetaInfoFilename, metaInfo)
prepLib.fuSavePepProbsTargetFromList('{!s}/target.csv'.format(out_strOutputBaseDir), listPepProb) 
