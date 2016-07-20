#!/usr/bin/env python3.5
# run parameters: CRUDDII/CRUDDII_ALL_n100_p91_th.txt CRUDDII/CRUDDII_ALL_n100_p91.1n.txt

import os
import sys
import numpy as np

AA=['I','L','V','F','M','C','A','G','P','T','S','Y','W','Q','N','H','E','D','K','R','B','X']
X=np.zeros((400,50104), dtype=np.int_)
X_sparse=np.empty((0,2), dtype=np.int_)
y=np.zeros(400, dtype=np.float)



# FILE 1=======================================================
f1=open(sys.argv[1],'r')
c=0
for line in f1:
#    print("c:" + str(c))
    line=line[:-1]
    for i in range(0,len(line)):
        if line[i] == 'X':
            X[c,i]=1
            X_sparse = np.append(X_sparse, np.array([[c, i]]), axis=0)
    #X[c,AA.index(line[i]),i]=1
    c=c+1

# FILE 2=======================================================
f2=open(sys.argv[2],'r')
c=0
for line in f2:
    y[c]=float(line[:-1])
    c=c+1

#print X
np.savetxt("input.csv", X, fmt="%1d", delimiter=",")
np.savetxt("input_sparse.csv", X_sparse, fmt="%1d", delimiter=",")
np.savetxt("target.csv", y, delimiter=",")

