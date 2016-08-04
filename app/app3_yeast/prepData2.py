import xml.etree.ElementTree as ET
import statistics as stat
import csv
import os
import re
import sys

from multiprocessing.dummy import Pool as ThreadPool

def fuSaveProtInfo(strLine1, strLine2, strBaseDir):
    if not strLine1 or not strLine2:
        return

    strProtName = strLine1[1:-1].split(' ')[0]
    strFilePath = '{!s}/{!s}.txt'.format(strBaseDir, strProtName)
    with open(strFilePath, 'w') as bfProt:
            bfProt.write(strLine2.strip())
    
    return

def fuPrepRefs(strFaPath, strBaseSaveDirPath):
    with open(strFaPath, 'r') as bfFasta:
        strLine1 = 'x'
        while strLine1:
            strLine1 = bfFasta.readline()
            strLine2 = bfFasta.readline()
            fuSaveProtInfo(strLine1, strLine2, strBaseSaveDirPath)

def fuLoadProtProbsFromPepXml(strXmlPath):
    tree = ET.parse(strXmlPath)
    root = tree.getroot()
    strBaseXmlAddr = "{http://regis-web.systemsbiology.net/pepXML}"

    # a) read all peptides and their probabilities:
    dicAll = {}
    for eSearchHit in root.findall(".//" +  strBaseXmlAddr + "search_hit"):
        strPeptide = eSearchHit.get('peptide')

        ePeptideProphetRes = eSearchHit.find(".//" + strBaseXmlAddr + "peptideprophet_result")
        dProb = ePeptideProphetRes.get('probability')

        if strPeptide not in dicAll:
            dicAll[strPeptide] = list()

        dicAll[strPeptide].append(float(dProb))

    # b) keep only one record for each peptide (average) ToDo: investigate why there are several records!
    for strPeptide, listProb in dicAll.items():
        dicAll[strPeptide] = stat.median(listProb)

    return dicAll

def fuSavePepProbsFlat(dicPeptideProbs, strFilePath):
    with open(strFilePath, 'w') as bfFile:
        for strPeptide, dProb in dicPeptideProbs.items():
            bfFile.write('{!s},{:.6f}\n'.format(strPeptide , dProb))

    return

def fuLoadPepProbsFromCsv(strFilePath):
    listPeptideProb = []
    with open(strFilePath, "r") as bfCsv:
        csvReader = csv.reader(bfCsv, delimiter=',')
        for row in csvReader:
            listPeptideProb.append([row[0], float(row[1])])

    return listPeptideProb

def fuSavePepProbsTargetFromList(strFilePath, listPeptideProb):
    with open(strFilePath, 'w') as bfFile:
        for row in listPeptideProb:
            dProb = row[1]
            bfFile.write('{:.6f}\n'.format(dProb))

    return

def fuGetProtRefFileNames(strBaseProtRefsPath):
    listProtFileName = os.listdir(strBaseProtRefsPath)
    
    return listProtFileName

def fuFindOnes(strProtSeq, strPepSeq):
    listMatches = []
    for match in re.finditer(strPepSeq, strProtSeq):
        start, end = match.span()
        listMatches = listMatches + [[start, end-start]]
    
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


def fuSaveProtPepOnes(strDir, strProtFileName, listProtPepOnes):
    strFilePath = strDir + '/' + strProtFileName

    with open(strFilePath, 'w') as bfFile:
        for row in listProtPepOnes:
            if len(row)>2:
                print("#######Look:" + strProtFileName )
            bfFile.write('{:d}:'.format(row[0]) )
            for listRange in row[1]:
                bfFile.write('|{:d},{:d}'.format(listRange[0], listRange[1]))
   #             bfFile.write(",".join(map(str, row[1])))
            bfFile.write("\n")

def fuGetProtLength(strFilePath):
    with open(strFilePath, 'r') as bfFile:
        nLength = len(bfFile.readline())
        return nLength


def fuSaveMetaInfo(strBasePath, strMetaInfoFilename):
    listProtFiles = [i for i in  os.listdir(strBasePath) if i.endswith('.txt') ]
    with open(strMetaInfoFilename, 'w') as bfFile:
        for strProtFileName in listProtFiles:
            strFilePath = '{!s}/{!s}'.format(strBaseProtRefsPath, strProtFileName)
            nProtWidth = fuGetProtLength(strFilePath)
            bfFile.write('{!s},{:d}\n'.format(strProtFileName, nProtWidth))

strSparseDir = './sparseData2'

strFaPath = '/home/user/eetemame/data/protein/yeast/sc_SGD_0604.fasta'
strBaseProtRefsPath = '/home/user/eetemame/data/protein/yeast/protRefs'
#fuPrepRefs(strFaPath, strBaseProtRefsPath )


strXmlPath = '/home/user/eetemame/data/protein/yeast/all/interact.pep.xml'
#dicPeptideProbs = fuLoadProtProbsFromPepXml(strXmlPath)

strFlatFile = '/home/user/eetemame/data/protein/yeast/all/peptideProbs.csv'
#fuSavePepProbsFlat(dicPeptideProbs, strFlatFile)

listPeptideProb = fuLoadPepProbsFromCsv(strFlatFile)
listProtRefFileName = fuGetProtRefFileNames(strBaseProtRefsPath)
#fuSavePepProbsTargetFromList('{!s}/target.csv'.format(strSparseDir), listPeptideProb) 

strMetaInfoFilename = '{!s}/metaInfo.csv'.format(strSparseDir)
#fuSaveMetaInfo(strSparseDir, strMetaInfoFilename)

# keep the following in the same order due to dependencies
def fuRunProt(strProtFileName): 
    print("#")
    listProtPepOnes = fuFindPeptideMatch(strBaseProtRefsPath , strProtFileName, listPeptideProb)
    if len(listProtPepOnes) > 0:
        fuSaveProtPepOnes(strSparseDir, strProtFileName, listProtPepOnes)
        print("saved:" + strProtFileName)
        return 1
    else:
        return 0

def fuRunAllProt(listProtFileName):
    '''
    isSave = fuRunProt(listProtFileName[1])
    print(listProtRefFileName[1])
    print(isSave)
    '''
    pool = ThreadPool(24)
    res = pool.map(fuRunProt, listProtFileName)
    pool.close() 
    pool.join() 
    print(res)

#nStart = int(sys.argv[1])
#nEnd = int(sys.argv[2])
#print('Running {:d}:{:d}'.format(nStart, nEnd))
#fuRunAllProt(listProtRefFileName[nStart:nEnd])
#fuRunAllProt(['Q0297.txt'])
