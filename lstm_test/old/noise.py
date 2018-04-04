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
from keras.layers import ELU, PReLU, Activation

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
n_models = 20
n_evaliter = 100
msets = [None for j in range(n_models)]
accs = [0 for j in range(n_models)]
accs2 = [0 for j in range(n_models)]


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


def offset_slice(inputs):
    w = 530
    r = np.random.randint(inputs.shape[1] - w + 1)
    return inputs[:, r:r + w, :]

for j in range(n_models):

    mset = gen_model()
    msets[j] = " " #mset

    m_in = Input(shape=x[0][0].shape)
    m_t = Lambda(offset_slice)(m_in)
    m_t = GaussianNoiseAlways(np.std(x[0][0] / 70))(m_t) # how much noice to have????

    m_t = Conv1D(30, 10, padding='causal')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = MaxPooling1D(2)(m_t)
    m_t = Dropout(0.2)(m_t)

    m_t = Conv1D(30, 5, padding='causal')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = MaxPooling1D(2)(m_t)
    m_t = Dropout(0.2)(m_t)

    m_t = Conv1D(30, 5, padding='causal')(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = ELU()(m_t)
    m_t = MaxPooling1D(2)(m_t)
    m_t = Dropout(0.2)(m_t)

    m_t = Flatten()(m_t)
    m_t = Dense(50)(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = Activation('tanh')(m_t)
    m_t = Dense(20)(m_t)
    m_t = BatchNormalization()(m_t)
    m_t = Activation('tanh')(m_t)
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
                          batch_size=64, epochs=200, verbose=0)
            h = h.history

            p1 = np.zeros(y[i][val].shape)
            p2 = np.zeros(yt[i].shape)
            for i2 in range(n_evaliter):
                p1 += model.predict(x[i][val])
                p2 += model.predict(xt[i])

            p1 /= n_evaliter
            p2 /= n_evaliter

            acc += np.mean(np.equal(np.argmax(p1, axis=-1),
                                    np.argmax(y[i][val], axis=-1)))
            acc2 += np.mean(np.equal(np.argmax(p2, axis=-1),
                                     np.argmax(yt[i], axis=-1)))


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
