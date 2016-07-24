#!/usr/bin/env python3.5
# run parameters: CRUDDII/CRUDDII_ALL_n100_p91_th.txt CRUDDII/CRUDDII_ALL_n100_p91.1n.txt

import sys
import csv

def sparseWriteLineToFiles(strLine, bfList, lineId):
    currProtId=0
    currProtStart=0
    isCurrProtEmpty=True
    protNonEmpty={}
    for i in range(0,len(strLine)):
        if strLine[i] == 'B':
            isCurrProtEmpty=True
            currProtId=currProtId+1
            currProtStart=i+1
        elif strLine[i] == 'X':
            if isCurrProtEmpty:
                bfList[currProtId].write('{:d}:'.format(lineId))
                isCurrProtEmpty=False
                protNonEmpty[currProtId]=True
            
            offset=i-currProtStart
            bfList[currProtId].write('{:d},'.format(offset))


    for currProtId, value in protNonEmpty.items():
        bfList[currProtId].write('\n')

    return #End sparseWriteLineToFiles

strDirname='./sparseData/'
nProteins=182
bfList=[]
metaInfo=[]

#a) Create protein files
for i in range(0, nProteins):
    strFilename='p{:d}.sprs'.format(i)
    metaInfo.append([strFilename])
    bfCurr=open(strDirname + strFilename, 'w')
    bfList.append(bfCurr)

#b) Read from bfInput and write to bfList (protein files)
bfInput=open(sys.argv[1], 'r')
rId=0
for strLine in bfInput:
    sparseWriteLineToFiles(strLine, bfList, rId)
    rId=rId+1


#c) Calculate width of each protein using the first line
bfInput.seek(0)
nWidth=0
currProtId=0
currChar=bfInput.read(1)
while currChar != '\n':

    if currChar == 'B':
        metaInfo[currProtId].append(nWidth)
        nWidth=-1
        currProtId=currProtId+1

    nWidth=nWidth+1
    currChar=bfInput.read(1)

metaInfo[currProtId].append(nWidth)

#d) Save the metaInfo
with open(strDirname + 'metaInfo.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(metaInfo)
