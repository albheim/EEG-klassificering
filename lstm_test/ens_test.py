import sys
from datetime import datetime

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
print(device_lib.list_local_devices())

import models
import data
import util


x, y = data.load_single(cut=True)

splits = 5
n_subs = len(x)
n_models = 20


for i in range(n_subs):
    print("################ SUB {} ####################".format(i + 1 if i + 1 < 10 else i + 2))
    n = x[i].shape[0]
    acc = 0
    for tr, val in util.kfold(n, splits):
        pred = np.zeros(y[i][val].shape)
        for j in range(n_models):
            first_layer_nodes = np.random.randint(10, 60)
            second_layer_nodes = np.random.randint(5, 30)
            dropout_prob = np.random.ranf() * 0.75
            model = models.lstm_dense(x[0][0].shape,
                                      first_layer_nodes,
                                      second_layer_nodes,
                                      dropout_prob)

            print("first {}, second {}, dropout {}".format(first_layer_nodes,
                                                           second_layer_nodes,
                                                           dropout_prob))
            w_save = model.get_weights()
            avgacc = 0
            # reset to initial weights
            model.set_weights(w_save)

            # fit with next kfold data
            model.fit(x[i][tr], y[i][tr],
                      batch_size=64, epochs=50, verbose=0)

            pred += model.predict(x[i][val], verbose=0)

        pred /= n_models
        acc = np.mean(

        acc /= splits
        avgacc += acc

        print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

    avgacc /= n_subs
    print("avg accuracy over all subjects {}".format(avgacc))


