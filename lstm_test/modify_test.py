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
msets = []
accs = []

def gen_model():
    return {"l1_nodes": np.random.randint(10, 40),
            "l1_filter": np.random.randint(5, 30),
            "l1_dropout": np.random.ranf() * 0.75,
            "l1_maxpool": np.random.randint(1, 4),
            "l2_nodes": np.random.randint(5, 30),
            "l2_filter": np.random.randint(4, 20),
            "l2_dropout": np.random.ranf() * 0.75,
            "l2_maxpool": np.random.randint(1, 3),
            "l3_nodes": np.random.randint(1, 15),
            "l3_filter": np.random.randint(2, 15),
            "l3_dropout": np.random.ranf() * 0.75,
            "l3_maxpool": np.random.randint(1, 3),
            "dense_nodes": np.random.randint(5, 50)}


for j in range(n_models):

    mset = gen_model()
    msets.append(mset)

    model = Sequential()

    model.add(Conv1D(mset["l1_nodes"], mset["l1_filter"], padding='causal',
                     input_shape=x[0][0].shape, activation='relu'))
    model.add(MaxPooling1D(mset["l1_maxpool"]))
    model.add(Dropout(mset["l1_dropout"]))

    model.add(Conv1D(mset["l2_nodes"], mset["l2_filter"],
                     padding='causal', activation='relu'))
    model.add(MaxPooling1D(mset["l2_maxpool"]))
    model.add(Dropout(mset["l2_dropout"]))

    model.add(Conv1D(mset["l3_nodes"], mset["l3_filter"],
                     padding='causal', activation='relu'))
    model.add(MaxPooling1D(mset["l3_maxpool"]))
    model.add(Dropout(mset["l3_dropout"]))

    model.add(Flatten())
    model.add(Dense(mset["dense_nodes"], activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    print(mset)

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
    accs.append(avgacc)
    print("avg accuracy over all subjects {}".format(avgacc))

for a, m in sorted(zip(accs, msets)):
    print("{}, {}".format(a, m))
