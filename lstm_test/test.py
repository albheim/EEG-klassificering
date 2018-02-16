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
import util


x, y = data.load_single()

splits = 5
n_subs = len(x)

mod = []
for i in range(20):
    mod.append(models.lstm_lstm(x[0][0].shape,
                                np.random.randint(10, 90),
                                np.random.randint(5, 50),
                                np.random.ranf() * 0.5 + 0.25))



for model in mod:
    model.summary()
    w_save = model.get_weights()
    avgacc = 0
    for i in range(n_subs):
        n = x[i].shape[0]
        acc = 0
        for tr, val in util.kfold(n, splits):
            # reset to initial weights
            model.set_weights(w_save)

            # fit with next kfold data
            model.fit(x[i][tr], y[i][tr],
                      batch_size=64, epochs=50, verbose=0)

            loss, accuracy = model.evaluate(x[i][val], y[i][val], verbose=0)
            acc += accuracy

        acc /= 10.0
        avgacc += acc

        print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

    avgacc /= n_subs
    print("avg accuracy over all subjects {}".format(avgacc))


