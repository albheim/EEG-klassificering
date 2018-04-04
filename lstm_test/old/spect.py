import sys
from datetime import datetime

import numpy as np
from scipy import io

import tensorflow as tf
import keras

from keras.models import Sequential, Model
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


x, y = data.load_spect_downsample(visual=True, ds=8)
# for i in range(len(x)):
#     x[i] = np.swapaxes(x[i], 0, 1)
#     print(x[i].shape)
#     x[i] = np.concatenate(x[i], axis=2)
#     print(x[i].shape)
print(x[0].shape)

splits = 5
n_subs = len(x)
n_models = 100


for _ in range(n_models):

    spect_in = Input(shape=x[0][0].shape)

    spect_t = Conv2D(32, (7, 2), activation='relu',
                     data_format='channels_first')(spect_in)
    spect_t = MaxPooling2D((2, 1))(spect_t)
    spect_t = Dropout(0.3)(spect_t)

    spect_t = Conv2D(16, (5, 3), activation='relu',
                     data_format='channels_first')(spect_t)
    spect_t = MaxPooling2D((2, 1))(spect_t)
    spect_t = Dropout(0.2)(spect_t)

    spect_t = Conv2D(16, (3, 1), activation='relu',
                     data_format='channels_first')(spect_t)

    spect_t = Flatten()(spect_t)
    spect_t = Dropout(0.3)(spect_t)
    spect_t = Dense(50, activation='tanh')(spect_t)

    out = Dense(3, activation="softmax")(spect_t)

    model = Model(inputs=spect_in, outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

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
                      validation_data=(x[i][val], y[i][val]),
                      batch_size=32, epochs=200, verbose=1)

            loss, accuracy = model.evaluate(x[i][val], y[i][val],
                                            verbose=0)
            acc += accuracy

        acc /= splits
        avgacc += acc

        print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, acc, splits))

    avgacc /= n_subs
    print("avg accuracy over all subjects {}".format(avgacc))
