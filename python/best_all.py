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
from keras import regularizers as rg
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import data
import util


x, y = data.load_single(cut=True, visual=True, study=True, transpose=True)
xt, yt = data.load_single(cut=True, visual=True, study=False, transpose=True)
print(x[0].shape, xt[0].shape)

x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)
xt = np.concatenate(xt, axis=0)
yt = np.concatenate(yt, axis=0)
print(x.shape, xt.shape)

splits = 10

def offset_slice(inputs):
    w = 630
    r = np.random.randint(inputs.shape[1] - w + 1)
    return inputs[:, r:r + w, :]


m_in = Input(shape=x[0].shape)
m_off = Lambda(offset_slice)(m_in)
m_noise = GaussianNoise(np.std(x[0] / 100))(m_off) # how much noice to have????

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
model.summary()

n = x.shape[0]
acc = 0
acc2 = 0
for tr, val in util.kfold(n, splits, shuffle=True):
    # recreate model
    model = Model.from_config(m_save)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # fit with next kfold data
    h = model.fit(x[tr], y[tr],
                  # validation_data=(x[i][val], y[i][val]),
                  batch_size=64, epochs=200, verbose=0)

    _, a = model.evaluate(x[val], y[val], verbose=0)
    _, a2 = model.evaluate(xt, yt, verbose=0)
    acc += a
    acc2 += a2

K.clear_session()

acc /= splits
acc2 /= splits

print("avg accuracy {}/{} over {} splits".format(acc, acc2, splits))
