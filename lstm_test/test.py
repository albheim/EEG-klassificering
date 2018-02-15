import sys

import numpy as np
from scipy import io

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN, RNN, LSTM, GRU

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print("here comes devices list")
print(device_lib.list_local_devices())
print("finished with device list")

import models
import data


xtr, ytr = data.load_single()


def kfold_split(n, k):
    s = np.arange(n)
    np.random.shuffle(s)
    for a in range(k):
        val = s[int(n * a / k):int(n * (a + 1) / k)]
        tr = np.concatenate((s[:int(n * a / k)], s[int(n * (a + 1) / k):]))
        yield (tr, val)

splits = 10
n_subs = len(xtr)

model = models.lstm_lstm(xtr[0][0].shape)

# w_save = model.get_weights()

model.summary()

avgacc = [0 for i in range(n_subs)]

for i in range(n_subs):
    for train, val in kfold_split(xtr[i].shape[0], splits):
        # reset to initial weights
        # model.set_weights(w_save)
        # fit with next kfold data
        model.fit(xtr[i][train], ytr[i][train],
                  batch_size=64, epochs=50, verbose=0)

        loss, accuracy = model.evaluate(xtr[i][val], ytr[i][val], verbose=0)
        avgacc[i] += accuracy

    avgacc[i] /= splits
    print("sub: {}  acc: {}".format(i + 1 if i + 1 < 10 else i + 2, avgacc[i]))
print("model: {}   avgacc: {}".format(j, sum(avgacc) / n_subs))
