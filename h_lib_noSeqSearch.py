import sys
import csv
import os

def updateProtPepDic(dicProt, nPepId, strProtName):
    if strProtName not in dicProt:
        dicProt[strProtName] = {}
    
    if nPepId not in dicProt[strProtName]:
        dicProt[strProtName][nPepId] = 0
    
    dicProt[strProtName][nPepId] += 1

def getPepDic(YInfo):
    dicPep = {}

    counter = 0
    for row in YInfo:
        dicPep[row[0]] = counter
        counter += 1
    return dicPep


def getXInfo(YInfo, in_strPeptideFilename, cDelim = " ", nSeqColId = 1, nProtColId = 2):
    dicProt = {}
    dicPep = getPepDic(YInfo)

    with open(in_strPeptideFilename, "r") as bfCsv:
        csvReader = csv.reader(bfCsv, delimiter = cDelim, skipinitialspace=True)
        counter = 0
        for row in csvReader:
            strPep = row[nSeqColId]
            nPepId = dicPep[strPep]
            nameParts = row[nProtColId].split('|')
            strProtName = nameParts[0]
            if len(strProtName)<3:
                strProtName = nameParts[1]
            strProtName = strProtName + '.txt'

            updateProtPepDic(dicProt, nPepId, strProtName)
    
    listProt = []
    for key, value in dicProt.items():
        listProt.append([key, value])
    
    return listProt
