# DePos

Requirements 
* python3.4/python3.5 
* torch7
* git clone https://github.com/ameenetemady/MyCommon # includes some utility code
* git clone https://github.com/ameenetemady/depos #this repository
* Install SparseNN from [here](https://github.com/ameenetemady/SparseNN/)
* luarocks install cephes
* luarocks install csv

Running DePos consists of three main steps:
##Step 1 - Data Preparation (generating sparse input format)#
* Create a directory under app (example directory “app5_18mix” is used here)
* Copy file experiment_3_multiArch.lua, lSettings.lua, prepData2.py from app5_18mix and create three empty directories under (app5_18mix):
  1. sparseData2 
  2. protRefs 
  3. model
* Update prepData2.py with paths of the peptide identification and protein  database (.fasta file)  in in_strFastaFilename and in_strPeptideFilename variables.
* Run “python3.4 prepData2.py” (assuming necessary packages installed using pip including the “Bio” package)

##Step 2 - Training, Prediction#
* Update three variables in lSettings.lua: 
  1. “strBaseDir” to the directory path, 
  2. “strFilenameProtRef” to the protein reference file (for evaluating final performance) (and don’t for get to copy the file to sparseData2 directory), 
  3. nRows to the number of rows in sparseData/target.csv
* Run:  th experiment_3_multiArch.lua 24
* Run: python3.4 ../../getAUC.py
