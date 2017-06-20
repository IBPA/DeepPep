""" Functions used for data preparation step of DeepPep. """

from Bio import SeqIO
import os
import csv
import statistics as stat
import re
from multiprocessing.dummy import Pool as ThreadPool

# Description: break the fasta file into multiple proteins, only save the ones provided in protsDic
def breakFasta(strFastaFilename, strProtRefsDir, protsDic = None):

  # create dir if missing
    if not os.path.exists(strProtRefsDir):
        os.makedirs(strProtRefsDir)

  # read from fasta file and generate one file for each protein
    counter = 0
    for currRecord in SeqIO.parse(strFastaFilename, "fasta"):
        if (protsDic is None) or (currRecord.name in  protsDic): # means: if protsDic is provided, then verify
            strFilePath = '{!s}/{!s}.txt'.format(strProtRefsDir, currRecord.name )
            with open(strFilePath, 'w') as bfProt:
                bfProt.write(str(currRecord.seq))

            counter += 1
    print('read and generated {:d} files'.format(counter))

def loadPepProbsFromCsv(strFilePath, delimiter, pepId, probId):
    listPeptideProb = []
    with open(strFilePath, "r") as bfCsv:
        csvReader = csv.reader(bfCsv, delimiter = delimiter, skipinitialspace=True)
        counter = 0
        for row in csvReader:
            listPeptideProb.append([row[pepId], float(row[probId])])

    return listPeptideProb

def loadProtPeptideDic(strFilePath, delimiter = "\t", protColId = 1, pepColId = 0, probColId = 2):
    """ Description: Load protein, peptide, and probablity info into single dictionary
    Return: 
      1) Dictionary of proteins pointed to the corresponding peptides and probabilities
      2) Dictionary of peptides with associated probababilities """

    protDic = {}
    pepDic = {}

    with open(strFilePath, "r") as bfCsv:
        csvReader = csv.reader(bfCsv, delimiter = delimiter, skipinitialspace=True)
        for row in csvReader:
            strPep = row[pepColId]
            strProt = row[protColId]
            dProb = float(row[probColId])

            # update peptide info
            pepInfo = pepDic.get(strPep)
            if pepInfo is None:
                pepInfo = [len(pepDic), strPep, dProb]
                pepDic[strPep] = pepInfo
            else:
                pepInfo[2] = max(dProb, pepInfo[2]) # Note: allways using 'max', if mean need a little more work
            
            # update protInfo
            protInfo = protDic.get(strProt)
            if protInfo is None:
                protInfo = {}
                protDic[strProt] = protInfo

            protInfo[strPep] = pepInfo
        
        return protDic, pepDic

def getProtRefFileNames(strBaseProtRefsPath):
    listProtFileName = os.listdir(strBaseProtRefsPath)
    return listProtFileName

def searchOnePepInOneProt(strProtSeq, strPepSeq):
    """ Finds the  occurances of strPepSeq in strProtSeq.
    Returns:
        listMatches: list containing pairs of begin,end for matching locations.
    """

    listMatches = []
    for match in re.finditer(strPepSeq, strProtSeq):
        start, end = match.span()
        listMatches = listMatches + [[start, end-start]]
    
    return listMatches

def searchPepsInOneProt(strBaseProtRefsPath, strProtFileName, listPeptideProb):
    """ For each protein find the list of matching peptides and their locations.
    Returns:
      listOnes: a list with matching peptide locations.
    """

    strProtFileName = strBaseProtRefsPath + '/' + strProtFileName

    listOnes = []

    with open(strProtFileName, 'r') as bfProtFile:
        strProtSeq = bfProtFile.read().strip()

        for item in listPeptideProb:
            i = item[0]
            strPepSeq = item[1]
            listPeptideOnes = searchOnePepInOneProt(strProtSeq, strPepSeq)

            if listPeptideOnes and  len(listPeptideOnes) > 0:
                listOnes.append([i, listPeptideOnes])

    return listOnes

def saveProtMatches(strDir, strProtFileName, listProtPepOnes):
    strFilePath = strDir + '/' + strProtFileName

    with open(strFilePath, 'w') as bfFile:
        for row in listProtPepOnes:
            if len(row)>2:
                print("####### ******************************* Look:" + strProtFileName )
            bfFile.write('{:d}:'.format(row[0]) )
            for listRange in row[1]:
                bfFile.write('|{:d},{:d}'.format(listRange[0], listRange[1]))
            bfFile.write("\n")

def searchAll(listProtFileName, strBaseProtRefsPath, strSparseDir, protsDic):
    """ Search for all peptides in all corresponding protein matches """

    def fuSearchOne(strProtFileName):
      print("started: " + strProtFileName)
      # remove the trailing ".txt" from filename
      strProtName = strProtFileName[0:-4]
      listPeptideProb = protsDic[strProtName].values()

      listProtPepOnes = searchPepsInOneProt(strBaseProtRefsPath , strProtFileName, listPeptideProb)
      if len(listProtPepOnes) > 0:
          saveProtMatches(strSparseDir, strProtFileName, listProtPepOnes)
          print("saved:" + strProtFileName)
          return 1
      else:
          return 0

    '''    
    isSave = fuSearchOne(listProtFileName[1])
    print(listProtRefFileName[1])
    print(isSave)
    '''    
 
    print(listProtFileName)
    pool = ThreadPool(32)
    res = pool.map(fuSearchOne, listProtFileName)
    pool.close() 
    pool.join() 
    print(res)

def getProtLength(strFilePath):
    with open(strFilePath, 'r') as bfFile:
        nLength = len(bfFile.readline())
        return nLength

def saveMetaInfo(strBasePath, strMetaInfoFilename, strBaseProtRefsPath):
    listProtFiles = [i for i in  os.listdir(strBasePath) if i.endswith('.txt') ]
    with open(strMetaInfoFilename, 'w') as bfFile:
        for strProtFileName in listProtFiles:
            strFilePath = '{!s}/{!s}'.format(strBaseProtRefsPath, strProtFileName)
            nProtWidth = getProtLength(strFilePath)
            bfFile.write('{!s},{:d}\n'.format(strProtFileName, nProtWidth))

def savePepProbsTargetFromList(strFilePath, listPeptideProb):
    with open(strFilePath, 'w') as bfFile:
        for row in listPeptideProb:
            dProb = row[1]
            bfFile.write('{:.6f}\n'.format(dProb))

    return
