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


x, y = data.load_single(cut=True, visual=True, transpose=True)
print(x[0].shape)

splits = 5
n_subs = len(x)
n_models = 100
msets = [None for j in range(n_models)]
accs = [0 for j in range(n_models)]

def gen_model():
    return {"l1_nodes": 30, #np.random.randint(10, 40),
            "l1_filter": 10, #np.random.randint(5, 30),
            "l1_dropout": 0.2, #np.random.ranf() * 0.75,
            "l2_nodes": 20, #np.random.randint(5, 30),
            "l2_filter": 5, #np.random.randint(4, 20),
            "l2_dropout": 0.2, #np.random.ranf() * 0.75,
            "l3_nodes": 10, #np.random.randint(1, 15),
            "l3_filter": 5, #np.random.randint(2, 15),
            "l3_dropout": 0.001, #np.random.ranf() * 0.75,
            "dense_nodes": 30} #np.random.randint(5, 50)}


for j in range(n_models):

    mset = gen_model()
    msets[j] = mset

    model = Sequential()

    model.add(Conv1D(mset["l1_nodes"], mset["l1_filter"],
                     input_shape=x[0][0].shape, activation='relu',
                     padding='causal'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(mset["l1_dropout"]))

    model.add(Conv1D(mset["l2_nodes"], mset["l2_filter"], activation='relu',
                     padding='causal'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(mset["l2_dropout"]))

    model.add(Conv1D(mset["l3_nodes"], mset["l3_filter"], activation='relu',
                     padding='causal'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(mset["l3_dropout"]))

    model.add(Flatten())
    model.add(Dense(mset["dense_nodes"], activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if j == 0:
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

        print("subject {}, avg accuracy {} over {} splits with ds {}".format(i + 1 if i + 1 < 10 else i + 2,
                                                                             acc, splits, 2**(j / 20)))

        if j % 20 == 19:
            x[i] = x[i][:, :, ::2]

    avgacc /= n_subs
    accs[j] = avgacc
    print("avg accuracy over all subjects {} for downsampling {}".format(avgacc, 2**(j / 20)))


for a, (j, m) in sorted(zip(accs, enumerate(msets))):
    print("acc {}, downsample {}\n{}\n".format(a, 2**(j / 20), m))
