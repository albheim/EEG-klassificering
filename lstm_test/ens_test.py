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


x, y = data.load_single(cut=True)

print("x size bytes", x[0].nbytes * len(x))

splits = 4
n_subs = len(x)
n_models = 10


avgacc = 0

for i in range(n_subs):
    n = x[i].shape[0]
    acc = 0
    for tr, val in util.kfold(n, splits):

        xtr = x[i][tr]
        ytr = y[i][tr]
        xva = x[i][val]
        yva = y[i][val]

        pred = np.zeros((len(val), 3))

        for j in range(n_models):

            model = models.lstm_lstm(xtr[0].shape,
                                     60, 20, 0.1)

            model.fit(xtr, ytr,
                      batch_size=64, epochs=50, verbose=0)

            pred += model.predict(xva, verbose=0)

        pred /= n_models
        acc += np.mean(np.equal(np.argmax(pred, axis=-1),
                                np.argmax(yva, axis=-1)))

    acc /= splits
    avgacc += acc

    print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, avgacc, splits))

avgacc /= n_subs
print("avg accuracy over all subjects {}".format(avgacc))


