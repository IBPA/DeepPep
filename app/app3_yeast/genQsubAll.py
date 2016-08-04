import os

nStepSize = 100
nLast = 6700

strFormat = 'qsub -V -cwd -q flavor.q -S /bin/bash -N j{0}_{1} runOne.sh prepData2.py {0} {1}'
for i in range(0, nLast, nStepSize):
    print(strFormat.format(i, i + nStepSize))

print(strFormat.format(nLast, -1))
