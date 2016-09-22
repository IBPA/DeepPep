#!/bin/bash
for i in $(seq 0.1 0.1 0.9); do
  echo $i "=========="
#  i=0.6
  th experiment_1.lua $i
  python3.4 ../../getAUC.py
done
