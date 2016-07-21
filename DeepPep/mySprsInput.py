#!/usr/bin/env python3.5
# run parameters: CRUDDII/CRUDDII_ALL_n100_p91_th.txt CRUDDII/CRUDDII_ALL_n100_p91.1n.txt

import os
import sys
import numpy as np
import csv

#X=np.zeros((400,50104), dtype=np.int_)
def sparseWriteLineToFiles(strLine, bfList, lineId):
    currProtId=0
    currProtStart=0
    isCurrProtEmpty=True
    protNonEmpty={}
    for i in range(0,len(strLine)):
        if strLine[i] == 'B':
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

nProteins=182
bfList=[]
metaInfo=[]

# Create protein files
for i in range(0, nProteins):
    strFilename='sparseData/p{:d}.sprs'.format(i)
    metaInfo.append([strFilename])
    bfCurr=open(strFilename, 'w')
    bfList.append(bfCurr)

# Read from bfInput and write to bfList (protein files)
'''
bfInput=open(sys.argv[1], 'r')
rId=0
for strLine in bfInput:
    sparseWriteLineToFiles(strLine, bfList, rId)
    rId=rId+1
'''

# Record the meta info
bfInput.seek(0)
bfInput.readline()
#for item in metaInfo:
#    print(item[0])

