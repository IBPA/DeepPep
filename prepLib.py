from Bio import SeqIO
import os

def breakFasta(strFastaFilename, strProtRefsDir):
  # create dir if missing
  if not os.path.exists(strProtRefsDir):
        os.makedirs(strProtRefsDir)

  # read from fasta file and generate one file for each protein
  counter = 0
  for currRecord in SeqIO.parse(strFastaFilename, "fasta"):
    strFilePath = '{!s}/{!s}.txt'.format(strProtRefsDir, currRecord.name.split('|')[0])
    with open(strFilePath, 'w') as bfProt:
            bfProt.write(str(currRecord.seq))

    counter += 1


  print('read and generated {:d} files'.format(counter))
