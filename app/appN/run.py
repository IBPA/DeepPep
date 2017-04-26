import sys
import os
import runLib
# General setting
dicTrainPredArgs = {'nArchId': 1}

def getDefaultSetting(strDatabaseDir, dicTrainPredArgs):
    dicSetting = {
        "dicTrainPredArgs": dicTrainPredArgs,
        "strDatasetDir": strDatabaseDir,
        "strFastaDB": '{:s}/db.fasta'.format(strDatabaseDir),
        "strFilePathIdentification": '{:s}/identification.tsv'.format(strDatabaseDir),
        "strDirSparseData": '{:s}/sparseData'.format(strDatabaseDir),
        "strDirPrepDataTmp": '{:s}/protRefs'.format(strDatabaseDir),
        "strFilePathPredOutput": '{:s}/pred.tsv'.format(strDatabaseDir),
        "strFilePathProtRefList": '{:s}/ref.txt'.format(strDatabaseDir)
    }

    return dicSetting

def runOne(dicSetting):
    # (a) prepData
    runLib.cleanup(dicSetting)
    nRows = runLib.prepData(dicSetting)
    dicSetting['nRows'] = nRows

    # (b) train, predict
    runLib.trainPred(dicSetting)

    # (c) calculate AUC

# 18mix:
strDataDir = '{!s}/data/protein/18mix'.format(os.environ.get('HOME'))
runOne(getDefaultSetting(strDataDir, dicTrainPredArgs))
