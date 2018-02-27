import sys
from datetime import datetime

import numpy as np
from scipy import io

import tensorflow as tf

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

import matplotlib.pyplot as plt


x, y = data.load_single(cut=True, visual=True, transpose=True)
print(x[0].shape)

n_subs = len(x)



model = Sequential()

model.add(Conv1D(10, 20, input_shape=x[0][0].shape, activation='relu',
                 strides=1, padding='causal'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Conv1D(10, 5, activation='relu', strides=1, padding='causal'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Conv1D(10, 5, activation='relu', strides=1, padding='causal'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(30, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

w_save = model.get_weights()
avgacc = 0
wlist = np.zeros((19,), dtype=np.object)
for i in range(n_subs):
    n = x[i].shape[0]

    tr, val = util.kfold(n, 5)[0]

    # reset to initial weights
    model.set_weights(w_save)

    # fit with next kfold data
    model.fit(x[i][tr], y[i][tr],
              batch_size=64, epochs=90, verbose=0,
              validation_data=(x[i][val], y[i][val]))

    loss, acc = model.evaluate(x[i][val], y[i][val],
                               verbose=0)

    avgacc += acc

    print("subject {}, accuracy {}".format(i + 1 if i + 1 < 10 else i + 2, acc))

    # Pick first layer and skip bias, shape is (10, 31, 30)
    w0 = model.layers[0].get_weights()[0]
    print(w0.shape)
    wlist[i] = w0


io.savemat('out.mat', mdict={'out': wlist})

avgacc /= n_subs
print("avg accuracy over all subjects {}".format(avgacc))

