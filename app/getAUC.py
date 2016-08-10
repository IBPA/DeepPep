import numpy as np
from numpy import genfromtxt
from sklearn import metrics


my_data = genfromtxt('sparseData2/protInfo.csv', delimiter=',')
y = my_data[:, 2]
pred = my_data[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
print(metrics.auc(fpr, tpr))

'''
y = np.array([1, 1, 2, 2])
pred = np.array([0.01, 0.04, 0.035, 0.08])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
print(metrics.auc(fpr, tpr))
'''
