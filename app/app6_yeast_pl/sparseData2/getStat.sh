#!/bin/bash
awk -F, '{
	cmd = "/usr/bin/wc -l " $1 "|xargs" "| cut -d \\  -f 1";
	cmd | getline out; 

#	print out  "x"  $2  "="  out * $2;
	print out*$2
	close(cmd)
}' metaInfo.csv

