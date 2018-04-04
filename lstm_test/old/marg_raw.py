import numpy as np

import tensorflow as tf
import keras

from keras.models import Sequential, Model
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


x, y = data.load_single(cut=True, shuffle=False)
x2, _ = data.load_marg(shuffle=False)

for i in range(len(x)):
    s = np.arange(x[i].shape[0])
    np.random.shuffle(s)
    x[i] = x[i][s]
    y[i] = y[i][s]
    x2[i] = x2[i][s]

print(x[0].shape, x2[0].shape)

print("x size mb", x[0].nbytes * len(x) / 1000000)
print("x2 size mb", x2[0].nbytes * len(x2) / 1000000)

splits = 5
n_subs = len(x)


avgacc = 0

for _ in range(10):
    for i in range(n_subs):
        n = x[i].shape[0]
        acc = 0
        for tr, val in util.kfold(n, splits):

            xtr = x[i][tr]
            xtr2 = x2[i][tr]
            ytr = y[i][tr]
            xva = x[i][val]
            xva2 = x2[i][val]
            yva = y[i][val]

            raw_in = Input(shape=xtr[0].shape)
            raw_t = LSTM(32, return_sequences=True)(raw_in)
            raw_t = LSTM(8)(raw_t)
            raw_out = Dense(30, activation="tanh")(raw_t)

            marg_in = Input(shape=xtr2[0].shape)
            marg_t = LSTM(32, return_sequences=True)(marg_in)
            marg_t = LSTM(8)(marg_t)
            marg_out = Dense(15, activation="tanh")(marg_t)

            merge = keras.layers.concatenate([raw_out, marg_out])
            merge_t = Dense(50, activation="tanh")(merge)
            out = Dense(3, activation="softmax")(merge_t)

            model = Model(inputs=[raw_in, marg_in], outputs=out)

            model.compile(optimizer="rmsprop",
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

            # model.summary()

            model.fit([xtr, xtr2], ytr,
                      batch_size=32, epochs=50, verbose=0)

            loss, accuracy = model.evaluate([xva, xva2], yva,
                                            batch_size=32, verbose=0)
            acc += accuracy


            # model = models.lstm_lstm(xtr[0].shape,
            #                          80, 25, 0.4)

            # model.fit(xtr, ytr,
            #           batch_size=64, epochs=50, verbose=0)

            # pred = model.predict(xva, verbose=0)

            # model = models.lstm_lstm(xtr2[0].shape,
            #                          60, 15, 0.4)

            # model.fit(xtr2, ytr,
            #           batch_size=64, epochs=50, verbose=0)

            # pred += model.predict(xva2, verbose=0)

            # pred /= 2
            # acc += np.mean(np.equal(np.argmax(pred, axis=-1),
            #                         np.argmax(yva, axis=-1)))

        acc /= splits
        avgacc += acc

        print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

    avgacc /= n_subs
    print("avg accuracy over all subjects {}".format(avgacc))


