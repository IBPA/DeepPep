import numpy as np
from numpy import genfromtxt
from sklearn import metrics


my_data = genfromtxt('sparseData2/protInfo.csv', delimiter=',')
y = my_data[:, 2]
pred = my_data[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
print("AUC(roc):{:f}".format( metrics.auc(fpr, tpr)))

precision, recall, thresholds = metrics.precision_recall_curve(y, pred, pos_label=1)
print("AUC(pr):{:f}".format( metrics.auc(recall, precision)))
