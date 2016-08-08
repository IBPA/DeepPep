from Bio import SeqIO
import os
import csv
import statistics as stat
import re
from multiprocessing.dummy import Pool as ThreadPool

def breakFasta(strFastaFilename, strProtRefsDir):
  # create dir if missing
  if not os.path.exists(strProtRefsDir):
        os.makedirs(strProtRefsDir)

  # read from fasta file and generate one file for each protein
  counter = 0
  for currRecord in SeqIO.parse(strFastaFilename, "fasta"):
    strFilePath = '{!s}/{!s}.txt'.format(strProtRefsDir, currRecord.name.split('|')[0])
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

def consolidatePepProbs(listPeptideProb):
    dicAll = {}

    # a) load all into dic
    for row in listPeptideProb:
      strPeptide = row[0]
      dProb = row[1]

      if strPeptide not in dicAll:
          dicAll[strPeptide] = list()

      dicAll[strPeptide].append(float(dProb))


    # b) consolidate
    listPeptideProbRes = []
    for strPeptide, listProb in dicAll.items():
        listPeptideProbRes.append([strPeptide, stat.median(listProb)])

    return listPeptideProbRes

def getProtRefFileNames(strBaseProtRefsPath):
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
            bfFile.write("\n")

def fuRunAllProt(listProtFileName, strBaseProtRefsPath, strSparseDir, listPeptideProb):

    def fuRunProt(strProtFileName):
      print("#")
      listProtPepOnes = fuFindPeptideMatch(strBaseProtRefsPath , strProtFileName, listPeptideProb)
      if len(listProtPepOnes) > 0:
          fuSaveProtPepOnes(strSparseDir, strProtFileName, listProtPepOnes)
          print("saved:" + strProtFileName)
          return 1
      else:
          return 0

    '''
    isSave = fuRunProt(listProtFileName[1])
    print(listProtRefFileName[1])
    print(isSave)
    '''
 
    print(listProtFileName)
    pool = ThreadPool(24)
    res = pool.map(fuRunProt, listProtFileName)
    pool.close() 
    pool.join() 
    print(res)

def fuGetProtLength(strFilePath):
    with open(strFilePath, 'r') as bfFile:
        nLength = len(bfFile.readline())
        return nLength

def fuSaveMetaInfo(strBasePath, strMetaInfoFilename, strBaseProtRefsPath):
    listProtFiles = [i for i in  os.listdir(strBasePath) if i.endswith('.txt') ]
    with open(strMetaInfoFilename, 'w') as bfFile:
        for strProtFileName in listProtFiles:
            strFilePath = '{!s}/{!s}'.format(strBaseProtRefsPath, strProtFileName)
            nProtWidth = fuGetProtLength(strFilePath)
            bfFile.write('{!s},{:d}\n'.format(strProtFileName, nProtWidth))

def fuSavePepProbsTargetFromList(strFilePath, listPeptideProb):
    with open(strFilePath, 'w') as bfFile:
        for row in listPeptideProb:
            dProb = row[1]
            bfFile.write('{:.6f}\n'.format(dProb))

    return

