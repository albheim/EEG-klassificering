import sys

import numpy as np
from scipy import io

import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import Lambda, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, SimpleRNN, RNN, LSTM, GRU

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


seed = 7
snic_tmp = str(sys.argv[1])

x = None
y = None

names = ["FA", "LM", "OB"]

for sub in [i if i < 10 else i + 1 for i in range(1, 2)]:  # 19 is max
    for i in range(3):
        name = "Subj{:02}_CleanData_study_{}".format(sub, names[i])
        print(name)
        m = io.loadmat('{}/DATA/Visual/{}.mat'.format(snic_tmp, name))
        trials = m[name][0][0][2][0]
        labels = np.zeros((trials.shape[0], 3))
        labels[:, i] = 1
        if x is None:
            x = trials
            y = labels
        else:
            x = np.concatenate((x, trials), axis=0)
            y = np.concatenate((y, labels), axis=0)

n = x.shape[0]
tr = int(0.8 * n)

x = np.stack(x, axis=0)

s = np.arange(n)
np.random.shuffle(s)
x = x[s]
y = y[s]

xtr = x[:tr]
print(xtr.shape)
ytr = y[:tr]
print(ytr.shape)
xte = x[tr:]
print(xte.shape)
yte = y[tr:]
print(yte.shape)




def kfold_split(n, k):
    s = np.arange(n)
    np.random.shuffle(s)
    for a in range(k):
        val = s[int(n * a / k):int(n * (a + 1) / k)]
        tr = np.concatenate((s[:int(n * a / k)], s[int(n * (a + 1) / k):]))
        yield (tr, val)


n_models = 10
avgacc = [0] * n_models
first = True

for train, val in kfold_split(xtr.shape[0], n_models):

    for i in range(n_models):
        model = Sequential()
        model.add(CuDNNLSTM((i + 2) * 5, input_shape=xtr[0].shape))  # returns a sequence of vectors of dimension 32
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(3, activation='softmax'))
        if first:
            model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.fit(xtr[train], ytr[train],
                  batch_size=64, epochs=10, verbose=0,
                  validation_data=(xtr[val], ytr[val]))

        loss, accuracy = model.evaluate(xtr[val], ytr[val], verbose=0)
        avgacc[i] += accuracy

    first = False


print("\n".join(["{}, accuracy: {}".format((i + 2) * 5, avgacc[i] / n_models) for i in range(n_models)]))
