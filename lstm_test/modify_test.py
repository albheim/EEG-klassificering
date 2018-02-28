import sys
from datetime import datetime

import numpy as np
from scipy import io

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN, RNN, LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, Flatten

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import models
import data
import util

datasize = 2

x, y = data.load_single(cut=False, visual=True, transpose=True)
x, y = data.modify(x, y, datasize, nmult=1.0, displacement=10)
print(x[0].shape)

splits = 5
n_subs = len(x)
n_models = 100
msets = [None for j in range(n_models)]
accs = [0 for j in range(n_models)]


for j in range(n_models):

    model = Sequential()

    model.add(Conv1D(40, 10, input_shape=x[0][0].shape, activation='relu',
                     padding='causal'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    model.add(Conv1D(20, 8, activation='relu', padding='causal'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    model.add(Conv1D(10, 5, activation='relu', padding='causal'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    w_save = model.get_weights()


    avgacc = 0
    for i in range(n_subs):
        n = int(x[i].shape[0] / datasize)
        acc = 0
        for tr, val in util.kfold(n, splits):
            tr = np.concatenate((tr, range(n, datasize * n)))
            np.random.shuffle(tr)

            # reset to initial weights
            model.set_weights(w_save)

            # fit with next kfold data
            model.fit(x[i][tr], y[i][tr],
                      batch_size=64, epochs=90, verbose=0)

            loss, accuracy = model.evaluate(x[i][val], y[i][val],
                                            verbose=0)
            acc += accuracy

        acc /= splits
        avgacc += acc

        print("subject {}, avg accuracy {}".format(i + 1 if i + 1 < 10 else i + 2, acc))

    avgacc /= n_subs
    accs[j] = avgacc
    print("avg accuracy over all subjects {}".format(avgacc))

