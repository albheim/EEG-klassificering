import sys
from datetime import datetime

import numpy as np
from scipy import io

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, GaussianNoise, BatchNormalization
from keras.layers import TimeDistributed, Lambda, AlphaDropout
from keras.layers import AveragePooling2D, SeparableConv2D
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import ELU, PReLU, Activation, Reshape

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from layers import GaussianNoiseAlways

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import data
import util


x, y = data.load_single(cut=True, visual=True, transpose=True)
xt, yt = data.load_single(cut=True, visual=True, study=False, transpose=True)
print(x[0].shape)

splits = 10
n_subs = len(x)
n_models = 100
n_evaliter = 10
msets = [None for j in range(n_models)]
accs = [0 for j in range(n_models)]
accs2 = [0 for j in range(n_models)]



for j in range(n_models):

    T, C = x[0][0].shape
    F = 10
    N = 3

    m_in = Input(shape=(T, C))
    m_t = Reshape((1, T, C))(m_in)

    m_t = Conv2D(F, (64, 1), padding='same')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = SeparableConv2D(F, (1, C), padding='valid')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = Dropout(0.2, noise_shape=(None, F, 1, 1))(m_t)

    m_t = SeparableConv2D(F, (8, 1), padding='valid')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = AveragePooling2D((4, 1))(m_t)
    m_t = Dropout(0.2, noise_shape=(None, F, 1, 1))(m_t)

    m_t = SeparableConv2D(F, (8, 1), padding='valid')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = AveragePooling2D((4, 1))(m_t)
    m_t = Dropout(0.2, noise_shape=(None, F, 1, 1))(m_t)

    m_t = Flatten()(m_t)
    m_out = Dense(3, activation='softmax')(m_t)

    model = Model(inputs=m_in, outputs=m_out)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    if j == 0:
        model.summary()


    w_save = model.get_weights()
    avgacc = 0
    avgacc2 = 0
    for i in range(n_subs):
        n = x[i].shape[0]
        acc = 0
        acc2 = 0
        for tr, val in util.kfold(n, splits):
            # reset to initial weights
            model.set_weights(w_save)

            # fit with next kfold data
            h = model.fit(x[i][tr], y[i][tr],
                          batch_size=64, epochs=50, verbose=0)
            h = h.history


            _, a = model.evaluate(x[i][val], y[i][val], verbose=0)
            _, a2 = model.evaluate(xt[i], yt[i], verbose=0)

            acc += a
            acc2 += a2


        acc /= splits
        acc2 /= splits
        avgacc += acc
        avgacc2 += acc2

        print("subject {}, avg accuracy {}/{} over {} splits".format(i + 1 if i + 1 < 10 else i + 2,
                                                                     acc, acc2, splits))

    avgacc /= n_subs
    accs[j] = avgacc
    avgacc2 /= n_subs
    accs2[j] = avgacc2
    print("avg accuracy over all subjects {}/{}".format(avgacc, avgacc2))


for a, a2 in sorted(zip(accs, accs2)):
    print("acc {}/{}\n".format(a, a2))

print("avg over all trials and subjects {}/{}".format(sum(accs) / len(accs), sum(accs2) / len(accs2)))
