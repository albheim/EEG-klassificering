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


x, y = data.load_single(cut=True, visual=False)


def get_random_setting():
    third_layer = np.random.choice([True, False])
    third_layer_type = np.random.choice([LSTM, Dense])
    third_layer_nodes = np.random.randint(10, 50)
    third_layer_dropout = np.random.ranf() * 0.75


    second_layer_nodes = np.random.randint(10, 100)
    second_layer_dropout = np.random.ranf() * 0.75
    if third_layer and third_layer_type == LSTM:
        second_layer_return_seq = True
    else:
        second_layer_return_seq = False

    first_layer_nodes = np.random.randint(50, 500)
    first_layer_dropout = np.random.ranf() * 0.75
    first_layer_return_seq = True

    epochs = np.random.randint(10, 100)

    return {"first_layer_nodes": first_layer_nodes,
            "first_layer_dropout": first_layer_dropout,
            "first_layer_return_seq": first_layer_return_seq,
            "second_layer_nodes": second_layer_nodes,
            "second_layer_dropout": second_layer_dropout,
            "second_layer_return_seq": second_layer_return_seq,
            "third_layer": third_layer,
            "third_layer_type": third_layer_type,
            "third_layer_nodes": third_layer_nodes,
            "third_layer_dropout": third_layer_dropout,
            "epochs": epochs}

splits = 5
n_subs = len(x)
n_models = 70
msets = []
accs = []


for _ in range(n_models):
    mset = get_random_setting()

    model = Sequential()

    model.add(LSTM(mset["first_layer_nodes"], 
                   input_shape=x[0][0].shape, 
                   return_sequences=mset["first_layer_return_seq"]))
    model.add(Dropout(mset["first_layer_dropout"]))

    model.add(LSTM(mset["second_layer_nodes"],
                   return_sequences=mset["second_layer_return_seq"]))
    model.add(Dropout(mset["second_layer_dropout"]))

    if mset["third_layer"]:
        model.add(mset["third_layer_type"](mset["third_layer_nodes"]))
        model.add(Dropout(mset["third_layer_dropout"]))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            

    model.summary()
    print(mset)

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
                      batch_size=64, epochs=mset["epochs"], verbose=0)

            loss, accuracy = model.evaluate(x[i][val], y[i][val], verbose=0)
            acc += accuracy

        acc /= splits
        avgacc += acc

        print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

    avgacc /= n_subs
    print("avg accuracy over all subjects {}".format(avgacc))

    msets.append(mset)
    accs.append(avgacc)

for acc, mset in sorted(zip(accs, msets)):
    print(acc, mset, "\n")


