import sys
import csv
import os

sys.path.append('../../')
import h_lib

in_strFastaFilename = '{!s}/data/protein/yeast/sc_SGD_0604.fasta'.format(os.environ.get('HOME'))
in_strPeptideFilename = '{!s}/data/protein/yeast/Yeast_dataset_peptide_identification.txt'.format(os.environ.get('HOME'))
in_strProtRefsDir = '../app6_yeast_pl/protRefs' # for reuse, maybe should copy it here
out_strOutputBaseDir = './sparseData_h'
strXMatchProb_filename = out_strOutputBaseDir + '/' + 'XMatchProb.marshal'


YInfo = h_lib.getPeptides(in_strPeptideFilename, "\t", 0, 2)
###assuming proteins are already broken to individual files under in_strProtRefsDir
XMatchProb = h_lib.getYInfo(YInfo, in_strProtRefsDir, strXMatchProb_filename, True)
YMatchProbCount = h_lib.getPeptideProteinMatches(YInfo, XMatchProb)
h_lib.updateXMatchingProbabilities(XMatchProb, YMatchProbCount)
XPred = h_lib.getAccumulatedXMatchingProbabilities(XMatchProb)

with open(out_strOutputBaseDir + "/h.csv", "w") as bfFile:
    for row in XPred:
        bfFile.write('{!s},{:.6f}\n'.format(row[0], row[1]))
