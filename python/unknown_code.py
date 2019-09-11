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
n_models = 1
bin_size = 40
n_bins = 6
p_size = bin_size * n_bins

start = 700
end = 1600
last = end - p_size
timepoints = range(start, last, bin_size)
steps = len(timepoints)

heatmap = np.zeros((steps, steps))
scores = np.zeros((steps, n_models))


# create net
m_in = Input(shape=(p_size, 31))
m_noise = GaussianNoise(np.std(x[0][0] / 100))(m_in) # how much noice to have????

m_t = Conv1D(20, 64, padding='causal')(m_noise)
m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling1D(2)(m_t)
m_t = Dropout(0.2)(m_t)

m_t = Conv1D(10, 32, padding='causal')(m_t)
m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling1D(2)(m_t)
m_t = Dropout(0.3)(m_t)

m_t = Conv1D(7, 16, padding='causal')(m_t)
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
for mod in range(n_models):
    for i in range(n_subs):
        # get data for sub i
        xm = np.mean(x[i], axis=(0, 1))[np.newaxis, np.newaxis, :]
        xs = np.std(x[i], axis=(0, 1))[np.newaxis, np.newaxis, :]
        xtr = (x[i] - xm) / xs
        ytr = y[i]
        xte = (xt[i] - xm) / xs
        yte = yt[i]

        n = xtr.shape[0]

        for tr, val in util.kfold(n, splits, shuffle=True):
            xxt = xtr[tr]
            yyt = ytr[tr]
            xxv = xtr[val]
            yyv = ytr[val]
            for t in range(steps):
                # recreate model
                model = Model.from_config(m_save)
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                # train model for time t
                model.fit(xxt[:, timepoints[t]:timepoints[t]+p_size], yyt,
                          batch_size=8, epochs=50, verbose=0)

                _, a = model.evaluate(xxv[:, timepoints[t]:timepoints[t]+p_size], yyv, verbose=0)
                scores[t, mod] += a

                # for t2 in range(steps):
                #     # eval model on test data for time t2
                #     _, a = model.evaluate(xte[:, timepoints[t2]:timepoints[t2]+p_size], yte, verbose=0)
                #     heatmap[t, t2] += a

            K.clear_session()

# heatmap /= (n_subs * n_models * splits)
# print(heatmap)
# np.savetxt("timepoints2.csv", heatmap, delimiter=',')

scores /= (n_subs * splits)
print(scores)
np.savetxt("timepoints2_scores.csv", scores, delimiter=',')