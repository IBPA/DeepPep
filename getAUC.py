import numpy as np
import sys
from numpy import genfromtxt
from sklearn import metrics
import pandas as pd
from ggplot import *

strFilename = 'sparseData2/protInfo.csv'
if len(sys.argv) > 1:
    strFilename = sys.argv[1]

my_data = genfromtxt(strFilename, delimiter=',')

# calucate AUC(roc)
y = my_data[:, 2]
pred = my_data[:, 1]
pred = pred/np.max(pred) # not necessary, can be removed

fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
print("{:f}".format( metrics.auc(fpr, tpr)), end=";")

# save ROC data
roc_data = np.stack((fpr, tpr), axis=1)
np.savetxt("{}.roc".format(strFilename), roc_data)

#plot ROC:
#df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
#plot_out = ggplot(df, aes(x='fpr', y='tpr')) +\
#            geom_line() +\
#            geom_abline(linetype='dashed')
#plot_out.save("roc.pdf")

# calucate AUC(pr)
precision, recall, thresholds = metrics.precision_recall_curve(y, pred, pos_label=1)
print("{:f}".format( metrics.auc(recall, precision)), end=";")

#save PR data
pr_data = np.stack((precision, recall), axis=1)
np.savetxt("{}.pr".format(strFilename), pr_data)


# calculate f1 score
if len(sys.argv) > 2:
    my_data = my_data[np.argsort(my_data[:, 1])][::-1]
    nTop = int(sys.argv[2])
    nRows = my_data.shape[0]
    labels = np.zeros([nRows, 1])
    labels[0:nTop,:].fill(1)
    my_data = np.append(my_data, labels, 1)

    # positive:
    f1Score = metrics.f1_score(my_data[:, 2], my_data[:, 3], pos_label=1)
    print("{:f}".format( f1Score), end=";")

    # negative:
    f1Score = metrics.f1_score(my_data[:, 2], my_data[:, 3], pos_label=0)
    print("{:f}".format( f1Score), end=";")

print()

