#!/bin/bash
pythonCommand=python3.5

prot_res=h.csv
prot_refListFile=../sparseData4/yeast_reference.csv

$pythonCommand $HOME/mygithub/ProteinLP/getAUC.py $prot_refListFile $prot_res 
