import sys
from datetime import datetime

import numpy as np
from scipy import io

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN, RNN, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D, Flatten

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import models
import data
import util


x, y = data.load_single(cut=True, visual=True)
for i in range(len(x)):
    x[i] = x[i].reshape((x[i].shape[0], x[i].shape[1], x[i].shape[2], 1))
print(x[0].shape)
print(x[0][0].shape)

splits = 5
n_subs = len(x)
n_models = 70
msets = []
accs = []


for _ in range(n_models):

    model = Sequential()

    model.add(Conv2D(32, (1, 10), input_shape=x[0][0].shape, activation='relu'))
    model.add(MaxPooling2D((1, 3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(10, (1, 5), activation='relu'))
    model.add(MaxPooling2D((1, 3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(5, (31, 5), activation='relu'))
    model.add(MaxPooling2D((1, 2)))
    model.add(Flatten())
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

            loss, accuracy = model.evaluate(x[i][val], y[i][val],
                                            verbose=0)
            acc += accuracy

        acc /= splits
        avgacc += acc

        print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

    avgacc /= n_subs
    print("avg accuracy over all subjects {}".format(avgacc))
