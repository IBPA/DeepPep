import re
import csv
import statistics as stat
from multiprocessing.dummy import Pool as ThreadPool
import prepLib

def fuFindOnes(strProtSeq, strPepSeq):
    listMatches = []
    for match in re.finditer(strPepSeq, strProtSeq):
        start, end = match.span()
        listMatches = listMatches + [[start, end]]
    
    return listMatches

def fuFindPeptideMatch(strBaseProtRefsPath, strProtFileName, listPeptideProb):
    strProtFileName = strBaseProtRefsPath + '/' + strProtFileName

    listOnes = []

    with open(strProtFileName, 'r') as bfProtFile:
        strProtSeq = bfProtFile.read().strip()

        for i in range(0, len(listPeptideProb)):
            strPepSeq = listPeptideProb[i][0]
            listPeptideOnes = fuFindOnes(strProtSeq, strPepSeq)

            if listPeptideOnes and  len(listPeptideOnes) > 0:
                listOnes.append([i, listPeptideOnes])

    return listOnes

def getOneEdgeMatch(pepMatches, edges):

    eMatchesOne = []
    for pM in pepMatches:
        eL = pM[0]
        eR = pM[1]
        idxL = edges.index(eL)
        idxR = edges.index(eR)
        eMatchesOne += [[idxL, idxR]]

    return eMatchesOne

def getEdgeMatches(edges, flistProtPepOnes):
    eMatches = []
    for s in flistProtPepOnes:
        eMatchesOne = getOneEdgeMatch(s[1], edges)
        eMatches += [[ s[0], eMatchesOne]]

    return eMatches

def getEdges(lSegments):
    edges = {}
    for sLine in lSegments:
        for s in sLine[1]:
            sL = s[0]
            sR = s[1]
            edges[sL] = True
            edges[sR] = True

    return sorted(list(edges.keys()))

def fuFindPeptideMatch_CleavageSites(strBaseProtRefsPath , strProtFilename, listPeptideProb):
    flistProtPepOnes = fuFindPeptideMatch(strBaseProtRefsPath, strProtFilename, listPeptideProb)
    edges = getEdges(flistProtPepOnes)
    eMatches = getEdgeMatches(edges, flistProtPepOnes)

    return eMatches, len(edges)

def fuRunAllProt_CleavageSites(listProtFileName, strBaseProtRefsPath, strSparseDir, listPeptideProb):
    def fuRunProt(strProtFileName):
        print("#start:" + strProtFileName)
        listProtPepOnes, nEdges = fuFindPeptideMatch_CleavageSites(strBaseProtRefsPath , strProtFileName, listPeptideProb)
        if len(listProtPepOnes) > 0:
            prepLib.fuSaveProtPepOnes(strSparseDir, strProtFileName, listProtPepOnes)
            #print("saved:" + strProtFileName)
            return [strProtFileName, nEdges]

    if len(listProtFileName) < 2 : # for test
        fuRunProt(listProtFileName[0])
        return
    
    #print(listProtFileName)
    pool = ThreadPool(8)
    res = pool.map(fuRunProt, listProtFileName)
    pool.close() 
    pool.join() 
    
    return list(filter(None.__ne__, res))

def appendDetectabilitiesFromCsv(listPepProb, strFilePath, delimiter, pepId, probId):
    dicPeptideDetect = {}

    #load:
    with open(strFilePath, "r") as bfCsv:
        csvReader = csv.reader(bfCsv, delimiter = delimiter, skipinitialspace=True)
        counter = 0
        for row in csvReader:
            strPeptide = row[pepId]
            dDetectability = float(row[probId])

            if strPeptide not in dicPeptideDetect:
                dicPeptideDetect[strPeptide] = list()
            
            dicPeptideDetect[strPeptide].append(float(dDetectability))

    #consolidate:
    count = 0
    for row in listPepProb:
        strPeptide = row[0]
        dDetectabilityConsolidated = float(1.0)

        if strPeptide in dicPeptideDetect:
            dDetectabilityConsolidated = stat.mean(dicPeptideDetect[strPeptide])
            count = count + 1
    
        row.append(dDetectabilityConsolidated)

    return listPepProb


def fuSavePepProbsTargetFromList(strFilePath, listPeptideProb):
    with open(strFilePath, 'w') as bfFile:
        for row in listPeptideProb:
            dProb = row[1]
            bfFile.write('{:.6f}'.format(dProb))

            if len(row)>2:
                dDetectability = row[2]
                bfFile.write(',{:.6f}'.format(dDetectability))
            
            bfFile.write('\n')

    return
