import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN, RNN, LSTM, GRU

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import models
import data
import util


x, y = data.load_marg(shuffle=True)

print(x[0].shape)

print("x size mb", x[0].nbytes * len(x) / 1000000)

splits = 5
n_subs = len(x)


avgacc = 0

for i in range(n_subs):
    n = x[i].shape[0]
    acc = 0
    for tr, val in util.kfold(n, splits):

        xtr = x[i][tr]
        ytr = y[i][tr]
        xva = x[i][val]
        yva = y[i][val]

        model = models.lstm_lstm(xtr[0].shape,
                                 60, 15, 0.4)

        model.fit(xtr, ytr,
                  batch_size=64, epochs=50, verbose=0)

        loss, accuracy = model.evaluate(xva, yva,
                                        verbose=0)
        acc += accuracy


    acc /= splits
    avgacc += acc

    print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

avgacc /= n_subs
print("avg accuracy over all subjects {}".format(avgacc))


