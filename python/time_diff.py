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

# import data
x, y = data.load_single(cut=False, visual=True, transpose=True)
xt, yt = data.load_single(cut=False, visual=True, study=False, transpose=True)
print(x[0].shape, xt[0].shape)


#settings
splits = 10
n_subs = len(x)
n_models = 20
bin_size = 40
n_bins = 6
p_size = bin_size * n_bins

train_start = 768
train_end = 1536
train_last = train_end - p_size
train_tp = range(train_start, train_last, bin_size)
train_steps = len(train_tp)

pred_start = 512
pred_end = 2048
pred_last = pred_end - p_size
pred_tp = range(pred_start, pred_last, bin_size)
pred_steps = len(pred_tp)

heatmap = np.zeros((pred_steps, pred_steps))


# create net
m_in = Input(shape=(p_size, 31))
m_noise = GaussianNoise(np.std(x[0][0] / 100))(m_in) # how much noice to have????

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


# train
for j in range(n_models):
    avgacc = 0
    for i in range(n_subs):
        n = x[i].shape[0]
        acc = 0
        for tr, val in util.kfold(n, splits, shuffle=True):
            # recreate model
            model = Model.from_config(m_save)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # fit with next kfold data
            xtr = x[i][tr]
            ytr = y[i][tr]
            xva = x[i][val]
            yva = y[i][val]
            xte = xt[i]
            yte = yt[i]
            for k in range(10):
                for t in train_tp:
                    model.fit(xtr[:, t:t+p_size], ytr,
                              batch_size=8, epochs=2, verbose=0)

            a1 = []
            a2 = []
            for t in pred_tp:
                _, a = model.evaluate(xva[:, t:t+p_size], yva, verbose=0)
                a1.append(a)
                _, a = model.evaluate(xte[:, t:t+p_size], yte, verbose=0)
                a2.append(a)

            # do a1/a2 avg over subject before jultiply?
            a1 = np.array(a1).reshape((len(a1), 1))
            a2 = np.array(a2).reshape((len(a2), 1))
            heatmap += a1 * a2.T

            acc += np.max(a1)

        K.clear_session()

        acc /= splits
        print("subject {} avg of max acc over {} splits {}".format(i, splits, acc))
        avgacc += acc

    avgacc /= n_subs
    print("avg over subjects", avgacc)

print(heatmap)
np.savetxt("heatmap.txt", delimiter=',')


