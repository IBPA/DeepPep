DeepPep, is a **protein identification** software which uses deep-convolutional neural network to predict the protein set from a proteomics mixture, given the sequence universe of possible proteins and a target peptide profile.

### Dependencies
* [torch7](http://torch.ch/docs/getting-started.html)
* luarocks install cephes
* luarocks install csv
* [SparseNN](https://github.com/ameenetemady/SparseNN/)
* python3.4 or above
* [biopython](http://biopython.org/wiki/Download)



### Installation
```
git clone https://github.com/ameenetemady/MyCommon.git
git clone https://github.com/ameenetemady/DeepPep.git
```

### Running
* Step1: edit run.py, and set "strDataDir" to a directory containing your input files (with exact names):

  * ```identification.tsv```: tab-delimeted file:  **column1**: peptide, **column2**: protein name, **column3**: identification probability
  * ```db.fasta```: reference protein database in fasta format.

* Step2: ```python run.py```

Upon completion, ```pred.csv``` will contain the predicted protein identification probabilities.

### Benchmark Datasets
There are [7 example datasets](./data) (used for benchmarking in the paper). Each dataset is generated from MS/MS raw files using TPP pipeline.

### Citation
 M. Kim, A. Eetemadi, and I. Tagkopoulos, “DeepPep: deep proteome inference from peptide profiling”, PLoS Computational Biology (2017) *under review*

### Licence
See the LICENSE.txt file for license rights and limitations (Apache2.0).

### Aknowledgement
This work was supported by a grant from Mars, Inc. and NSF award 1516695.

