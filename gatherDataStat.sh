#!/bin/bash
strDir=$1
#total number of peptides
nPeptides=$(wc -l $strDir/target.csv |awk '{print $1}')

#number of simple peptides
#number of shared peptides
for protFileName in $(ls $strDir/*.txt); do

	for peptideInfo in $(cat $protFileName | awk -F'[,\|:]' '{print $1 "_" $4;}'); do
		nPeptideId=$(echo $peptideInfo | cut -d_ -f 1)
		nPeptideLength=$(echo $peptideInfo | cut -d_ -f 2)
		arrPepLength[$nPeptideId]=$nPeptideLength

		if [ -z ${arrPep[$nPeptideId]} ]; then
			arrPep[$nPeptideId]=1
		else
			arrPep[$nPeptideId]=$((arrPep[$nPeptideId] + 1))
		fi

	done
#	for nPeptideLength in  $(cat $protFileName | awk -F'[,|:]', '{print $2}' | awk -F\| '{print $1}');  do
#		echo $protFileName \* $nPeptideLength
#	done
done

nSimplePeptides=0
nSharedPeptides=0
for nPeptideCount in "${arrPep[@]}"; do
	if [ $nPeptideCount -eq 1 ]; then
		nSimplePeptides=$(($nSimplePeptides+1))
	else
		nSharedPeptides=$(($nSharedPeptides+1))
	fi
done


#avg peptide length
nSum=0
for nPeptideLength in "${arrPepLength[@]}"; do
	if [ ! -z $nPeptideLength ]; then
		nSum=$(($nSum + $nPeptideLength))
	fi
done

echo nSum: $nSum
nAvgPeptideLength=$(($nSum/$nPeptides))


#number of proteins
nCandidateProteins=$(ls $strDir/*.txt|wc -l|awk '{print $1}')

#average peptide probability
dAvgPepProb=$(awk '{ total += $1 } END { print total/NR }' $strDir/target.csv)
echo nPeptides: $nPeptides
echo nSimplePeptides: $nSimplePeptides
echo nSharedPeptides: $nSharedPeptides
echo nAvgPeptideLength: $nAvgPeptideLength
echo nCandidateProteins: $nCandidateProteins
echo dAvgPepProb: $dAvgPepProb
