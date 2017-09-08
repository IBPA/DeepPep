### What is DeepPep?

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
git clone https://github.com/DeepPep/DeepPep.git
```

### Running
* Step1: prepare a directory containing your input files (with exact names):

  * ```identification.tsv```: tab-delimeted file:  **column1**: peptide, **column2**: protein name, **column3**: identification probability
  * ```db.fasta```: reference protein database in fasta format.

* Step2: ```python run.py directoryName```

Upon completion, ```pred.csv``` will contain the predicted protein identification probabilities.

### Benchmark Datasets
There are [7 example datasets](https://github.com/DeepPep/public/tree/master/data) (used for benchmarking in the paper). Each dataset is generated from MS/MS raw files using TPP pipeline. For example, to run the [18Mix benchmark dataset](https://github.com/DeepPep/public/tree/master/data/18mix), simply run the following:

```
python run.py data/18Mix
```
### Support

If you have any questions about DeepPep, please contact Minseung Kim (msgkim@ucdavis.edu) or Ameen Eetemadi (eetemadi@ucdavis.edu).

### Citation
 M. Kim, A. Eetemadi, and I. Tagkopoulos, “DeepPep: deep proteome inference from peptide profiling”, PLoS Computational Biology (2017) [\[link\]](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005661)

### Licence
See the [LICENSE](./LICENSE) file for license rights and limitations (Apache2.0).

### Acknowledgement
This work was supported by a grant from Mars, Inc. and NSF award 1516695.

