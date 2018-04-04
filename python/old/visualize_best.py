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

from layers import GaussianNoiseAlways

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import data
import util


x, y = data.load_single(cut=True, visual=True, transpose=True)

# select a single channel
ch = 6
x = [x[i][:, :, ch:ch+1] for i in range(len(x))]
print(x[0].shape)

splits = 5
n_subs = len(x)
n_models = 20
accs = [0 for j in range(n_models)]


def offset_slice(inputs):
    w = 630
    r = np.random.randint(inputs.shape[1] - w + 1)
    return inputs[:, r:r + w, :]

wlist = np.zeros((19,), dtype=np.object)


m_in = Input(shape=x[0][0].shape)
m_off = Lambda(offset_slice)(m_in)
m_noise = GaussianNoise(np.std(x[0][0] / 100))(m_off) # how much noice to have????

m_t = Conv1D(5, 64, padding='causal')(m_noise)
#m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling1D(2)(m_t)
m_t = Dropout(0.2)(m_t)

m_t = Conv1D(10, 32, padding='causal')(m_t)
#m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling1D(2)(m_t)
m_t = Dropout(0.2)(m_t)

m_t = Conv1D(20, 16, padding='causal')(m_t)
#m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling1D(2)(m_t)
m_t = Dropout(0.2)(m_t)

m_t = Flatten()(m_t)
# m_t = Dense(50)(m_t)
# m_t = BatchNormalization()(m_t)
# m_t = Activation('tanh')(m_t)
m_t = Dense(20)(m_t)
m_t = BatchNormalization()(m_t)
m_t = Activation('tanh')(m_t)
m_out = Dense(3, activation='softmax')(m_t)

model = Model(inputs=m_in, outputs=m_out)

m_save = model.get_config()

model.summary()

avgacc = 0
wlist = np.zeros((18,), dtype=np.object)
for i in range(n_subs):
    n = x[i].shape[0]
    acc = 0

    model = Model.from_config(m_save)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    tr, val = util.kfold(n, splits)[0]


    # fit with next kfold data
    h = model.fit(x[i][tr], y[i][tr],
                  batch_size=64, epochs=50, verbose=0)

    _, acc = model.evaluate(x[i][val], y[i][val], verbose=0)

    print("subject {}, avg accuracy {}".format(i + 1 if i + 1 < 10 else i + 2, acc))
    layers = model.layers
    print(len(layers))
    print(layers[3])
    w0 = model.layers[3].get_weights()[0]
    print(w0.shape)
    wlist[i] = w0

io.savemat('out_ch{}.mat'.format(ch), mdict={'out': wlist})
