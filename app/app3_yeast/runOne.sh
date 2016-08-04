#!/bin/bash
strRunName=$1_$2_$3
cmd="/usr/bin/python3.4 $1 $2 $3 > result/$strRunName.out 2>&1 "
eval "$cmd" 
