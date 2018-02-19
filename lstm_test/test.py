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
print("here comes devices list")
print(device_lib.list_local_devices())
print("finished with device list")
print(str(datetime.now()))

import models
import data
import util


x, y = data.load_single(cut=True)


def get_random_setting():
    second_layer = np.random.choice([True, False])
    second_layer_type = np.random.choice([LSTM, Dense])
    second_layer_nodes = np.random.randint(10, 100)
    second_layer_dropout = np.random.ranf() * 0.5 + 0.25

    first_layer_nodes = np.random.randint(10, 100)
    first_layer_dropout = np.random.ranf() * 0.5 + 0.25
    if second_layer:
        first_layer_return_seq = np.random.choice([True, False])
    else:
        first_layer_return_seq = False

    output = np.random.choice([Dense])

    epochs = np.random.randint(10, 100)

    return {"first_layer_nodes": first_layer_nodes,
            "first_layer_dropout": first_layer_dropout,
            "first_layer_return_seq": first_layer_return_seq,
            "second_layer": second_layer,
            "second_layer_type": second_layer_type,
            "second_layer_nodes": second_layer_nodes,
            "second_layer_dropout": second_layer_dropout,
            "output_type": output,
            "epochs": epochs}

splits = 5
n_subs = len(x)
n_models = 20


for _ in range(n_models):
    first_layer_nodes = np.random.randint(10, 60)
    second_layer_nodes = np.random.randint(5, 30)
    dropout_prob = np.random.ranf() * 0.75
    model = models.lstm_dense(x[0][0].shape,
                              first_layer_nodes,
                              second_layer_nodes,
                              dropout_prob)

    model.summary()
    print("first {}, second {}, dropout {}".format(first_layer_nodes,
                                                   second_layer_nodes,
                                                   dropout_prob))
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

        acc /= splits
        avgacc += acc

        print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

    avgacc /= n_subs
    print("avg accuracy over all subjects {}".format(avgacc))


