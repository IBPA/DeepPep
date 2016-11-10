import numpy as np
import sys
from numpy import genfromtxt
from sklearn import metrics

strFilename = 'sparseData2/protInfo.csv'
if len(sys.argv) > 1:
    strFilename = sys.argv[1]

my_data = genfromtxt(strFilename, delimiter=',')
y = my_data[:, 2]
pred = my_data[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
#print("AUC(roc):{:f}".format( metrics.auc(fpr, tpr)), end=";")
print("{:f}".format( metrics.auc(fpr, tpr)), end=";")

precision, recall, thresholds = metrics.precision_recall_curve(y, pred, pos_label=1)
#print("AUC(pr):{:f}".format( metrics.auc(recall, precision)))
print("{:f}".format( metrics.auc(recall, precision)))
