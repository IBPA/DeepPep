# ** imports:

import sys
import marshal
import os
import re
import csv
import statistics as stat
from multiprocessing.dummy import Pool as ThreadPool

sys.path.append('../..')
import prepLib

#** private methods:

def fuGetOnePeptideMatchCount(strProtSeq, strPepSeq):
    count = 0
    for match in re.finditer(strPepSeq, strProtSeq):
        count = count + 1
    
    return count

def fuGetOneProteinPeptideMatches(strBaseProtRefsPath , strProtFileName, listPeptideProb):
    strProtFileName = strBaseProtRefsPath + '/' + strProtFileName

    dicRes = {}

    with open(strProtFileName, 'r') as bfProtFile:
        strProtSeq = bfProtFile.read().strip()

        for i in range(0, len(listPeptideProb)):
            strPepSeq = listPeptideProb[i][0]
            matchCount = fuGetOnePeptideMatchCount(strProtSeq, strPepSeq)

            if matchCount > 0:
                dicRes[i] = matchCount

    return dicRes    

def fuRunAllProt(listProtFileName, strBaseProtRefsPath, listPeptideProb):

    def fuRunProt(strProtFileName):
        print("#start:" + strProtFileName)
        dicProtPepMatches = fuGetOneProteinPeptideMatches(strBaseProtRefsPath , strProtFileName, listPeptideProb)
        if len(dicProtPepMatches) > 0:
            return [strProtFileName, dicProtPepMatches]

    if len(listProtFileName) < 1 : # for test
        fuRunProt(listProtFileName[0])
        return
    
    pool = ThreadPool(8)
    res = pool.map(fuRunProt, listProtFileName)
    pool.close() 
    pool.join() 
    
    return list(filter(None.__ne__, res))

#** public methods:
def getPeptides(in_strPeptideFilename, cDelim = " ", nSeqColId = 1, nProbColId = 3):
    listPepProb = prepLib.loadPepProbsFromCsv(in_strPeptideFilename, cDelim, nSeqColId, nProbColId)
    listPepProb = prepLib.consolidatePepProbs(listPepProb)
    return listPepProb

def getProteinPeptideMatches(listPepProb, in_strProtRefsDir):
    listProtRefFileName = prepLib.getProtRefFileNames(in_strProtRefsDir)
    #listProtRefFileName = ['P06396.txt', 'IPI00025499.1.txt']
    res = fuRunAllProt(listProtRefFileName, in_strProtRefsDir, listPepProb)

    return res

def getYInfo(YInfo, in_strProtRefsDir, strXMatchProb_filename, isRedo=False):
    XMatchProb = None
    if isRedo:
        with open(strXMatchProb_filename, 'wb') as f:
            XMatchProb = getProteinPeptideMatches(YInfo, in_strProtRefsDir)
            marshal.dump(XMatchProb, f)
    else:
        with open(strXMatchProb_filename, 'rb') as f:
            XMatchProb = marshal.load(f)
    
    return XMatchProb

def getPeptideProteinMatches(listPepProb, XMatchProb):
    for protInfo in XMatchProb:
        for nPeptideId, nMatchCount in protInfo[1].items():
            peptideInfo = listPepProb[nPeptideId]
            
            #ensure the peptide has a count column
            if len(peptideInfo) < 3:
                peptideInfo.append(0)
            
            #increment
            peptideInfo[2] += nMatchCount

    return listPepProb

# calculate each match's share and assign the probability
def updateXMatchingProbabilities(XMatchProb, YMatchProbCount):
    for protInfo in XMatchProb:
        for nPeptideId, nMatchCount in protInfo[1].items():
            dProb = YMatchProbCount[nPeptideId][1]/YMatchProbCount[nPeptideId][2]
            protInfo[1][nPeptideId] = [nMatchCount, dProb]

# sum all the probabilities in each protein
def getAccumulatedXMatchingProbabilities(XMatchProb):
    XPred = []
    for protInfo in XMatchProb:
        strProtName = protInfo[0][:-4]
        dPred = 0
        for key, value in protInfo[1].items():
            dPred += value[0]*value[1]
        
        XPred.append([strProtName, dPred])

    return XPred