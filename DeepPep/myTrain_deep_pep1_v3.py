#!/usr/bin/env python3.5
# run parameters: CRUDDII/CRUDDII_ALL_n100_p91_th.txt CRUDDII/CRUDDII_ALL_n100_p91.1n.txt

import os
import sys
import numpy as np
import theano
import lasagne
import nolearn
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
		epoch = train_history[-1]['epoch']
		new_value = np.float32(self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv1DLayer),
        ('pool2', layers.MaxPool1DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv1DLayer),
        ('pool3', layers.MaxPool1DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 50104),
    conv1_num_filters=20, conv1_filter_size=10,
    pool1_pool_size=4,
    dropout1_p=0.2,
    conv2_num_filters=60, conv2_filter_size=10,
    pool2_pool_size=5,
    dropout2_p=0.2,
    conv3_num_filters=100, conv3_filter_size=10,
    pool3_pool_size=4,
    dropout3_p=0.2,
    hidden4_num_units=500,#hidden4_nonlinearity=lasagne.nonlinearities.sigmoid,
    dropout4_p=0.2,
    hidden5_num_units=500,
    output_num_units=1, output_nonlinearity=None,#lasagne.nonlinearities.sigmoid,

    #update_learning_rate=0.01,
    #update_momentum=0.9,

	update_learning_rate=theano.shared(np.float32(0.03)),
	update_momentum=theano.shared(np.float32(0.9)),
	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
	],

    regression=True,
    max_epochs=100,
    verbose=1,
    )


# X=np.random.random((400,22,5705)).astype(np.float32)
# y=np.random.random(400).astype(np.float32)
AA=['I','L','V','F','M','C','A','G','P','T','S','Y','W','Q','N','H','E','D','K','R','B','X']
X=np.zeros((400,1,50104))
y=np.zeros(400)
V=np.zeros((400,1,50104))
# FILE 1=======================================================
f1=open(sys.argv[1],'r')
c=0
for line in f1:
	line=line[:-1]
	for i in range(0,len(line)):
		if line[i] == 'X':
			X[c,0,i]=1
        #X[c,AA.index(line[i]),i]=1
	c=c+1
# FILE 2=======================================================
f2=open(sys.argv[2],'r')
c=0
for line in f2:
    y[c]=float(line[:-1])
    c=c+1
X=X.astype(np.float32)
y=y.astype(np.float32)
#net2.fit(X, y)

