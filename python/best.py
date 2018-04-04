import sys
from datetime import datetime

import numpy as np
from scipy import io

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, GaussianNoise, BatchNormalization
from keras.layers import TimeDistributed, Lambda, AlphaDropout
from keras.layers import SimpleRNN, RNN, LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import ELU, PReLU, Activation, AveragePooling1D

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import data
import util


x, y = data.load_single(cut=True, visual=False, transpose=True)
xt, yt = data.load_single(cut=True, visual=False, study=False, transpose=True)
print(x[0].shape, xt[0].shape)

splits = 10
n_subs = len(x)
n_models = 20
msets = [None for j in range(n_models)]
accs = [0 for j in range(n_models)]
accs2 = [0 for j in range(n_models)]

# channels = [4, 23]
# for i in range(n_subs):
#     x[i] = x[i][:, :, channels]
#     xt[i] = xt[i][:, :, channels]


def offset_slice(inputs):
    w = 630
    r = np.random.randint(inputs.shape[1] - w + 1)
    return inputs[:, r:r + w, :]

for j in range(n_models):

    msets[j] = " " # mset

    m_in = Input(shape=x[0][0].shape)
    m_off = Lambda(offset_slice)(m_in)
    m_noise = GaussianNoise(np.std(x[0][0] / 100))(m_off) # how much noice to have????

    m_t = Conv1D(30, 64, padding='causal')(m_noise)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = AveragePooling1D(2)(m_t)
    m_t = Dropout(0.2)(m_t)

    m_t = Conv1D(15, 32, padding='causal')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = AveragePooling1D(2)(m_t)
    m_t = Dropout(0.3)(m_t)

    m_t = Conv1D(10, 16, padding='causal')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = AveragePooling1D(2)(m_t)
    m_t = Dropout(0.4)(m_t)

    m_t = Flatten()(m_t)
    # m_t = Dense(35)(m_t)
    # m_t = BatchNormalization()(m_t)
    # m_t = Activation('tanh')(m_t)
    m_t = Dense(15)(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = Activation('tanh')(m_t)
    m_out = Dense(3, activation='softmax')(m_t)

    model = Model(inputs=m_in, outputs=m_out)

    m_save = model.get_config()
    if j == 0:
        model.summary()

    avgacc = 0
    avgacc2 = 0
    for i in range(n_subs):
        n = x[i].shape[0]
        acc = 0
        acc2 = 0
        for tr, val in util.kfold(n, splits, shuffle=True):
            # recreate model
            model = Model.from_config(m_save)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # fit with next kfold data
            h = model.fit(x[i][tr], y[i][tr],
                          # validation_data=(x[i][val], y[i][val]),
                          batch_size=64, epochs=200, verbose=0)
            h = h.history

            _, a = model.evaluate(x[i][val], y[i][val], verbose=0)
            _, a2 = model.evaluate(xt[i], yt[i], verbose=0)
            acc += a
            acc2 += a2

        K.clear_session()

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
