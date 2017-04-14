# same as h.py, but using the identification file only and no seq search
import sys
import csv
import os

sys.path.append('../../')
import h_lib
import h_lib_noSeqSearch

in_strPeptideFilename = '{!s}/data/protein/sigma_49/Sigma_49.txt'.format(os.environ.get('HOME'))
out_strOutputBaseDir = './sparseData_h'
out_strFile = out_strOutputBaseDir + "/h_noSeqSearch.csv"

YInfo = h_lib.getPeptides(in_strPeptideFilename)
XMatchProb = h_lib_noSeqSearch.getXInfo(YInfo, in_strPeptideFilename)

# rest: same as in h:
YMatchProbCount = h_lib.getPeptideProteinMatches(YInfo, XMatchProb)
h_lib.updateXMatchingProbabilities(XMatchProb, YMatchProbCount)
XPred = h_lib.getAccumulatedXMatchingProbabilities(XMatchProb)

XPred.sort()
with open(out_strFile, "w") as bfFile:
    for row in XPred:
        bfFile.write('{!s},{:.6f}\n'.format(row[0], row[1]))

print("result saved in:" + out_strFile)
