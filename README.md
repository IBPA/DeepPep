# DePos

There are two main requirements 
* python3.4/python3.5 
* torch7

Running DePos consists of three main steps:
##Step 1 - Data Preparation (generating sparse input format)#
* Create a directory under app (example directory “app5_18mix” is used here)
* Copy file experiment_1.lua, lSettings.lua from app5_18mix and create three empty directories under (app5_18mix):
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
* Run:  th experiment_1.lua
* Run: python3.4 ../../getAUC.py
