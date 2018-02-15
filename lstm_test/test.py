import sys

import numpy as np
from scipy import io

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN, RNN, LSTM, GRU

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print("here comes devices list")
print(device_lib.list_local_devices())
print("finished with device list")



snic_tmp = "C:/Users/Albin Heimerson/Desktop/exjobb/"
if len(sys.argv) > 1:
    snic_tmp = str(sys.argv[1])

x = []
y = []

names = ["FA", "LM", "OB"]

for sub in [i if i < 10 else i + 1 for i in range(1, 19)]:  # 19 is max
    xn = None
    yn = None
    for i in range(3):
        name = "Subj{:02}_CleanData_study_{}".format(sub, names[i])
        print(name)
        m = io.loadmat('{}/DATA/Visual/{}.mat'.format(snic_tmp, name))
        trials = m[name][0][0][2][0]
        for j in range(trials.shape[0]):
            trials[j] = trials[j][:, 768:1536]
        labels = np.zeros((trials.shape[0], 3))
        labels[:, i] = 1
        if xn is None:
            xn = trials
            yn = labels
        else:
            xn = np.concatenate((xn, trials), axis=0)
            yn = np.concatenate((yn, labels), axis=0)

    xn = np.stack(xn, axis=0)
    n = xn.shape[0]
    s = np.arange(n)
    np.random.shuffle(s)
    x.append(xn[s])
    y.append(yn[s])

xtr = x
ytr = y

def kfold_split(n, k):
    s = np.arange(n)
    np.random.shuffle(s)
    for a in range(k):
        val = s[int(n * a / k):int(n * (a + 1) / k)]
        tr = np.concatenate((s[:int(n * a / k)], s[int(n * (a + 1) / k):]))
        yield (tr, val)

splits = 10
n_subs = len(xtr)

model = Sequential()
model.add(LSTM(32, input_shape=xtr[0][0].shape, return_sequences=True))
model.add(Dropout(0.5))
# model.add(LSTM(16))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# w_save = model.get_weights()

model.summary()

avgacc = [0 for i in range(n_subs)]

for i in range(n_subs):
    for train, val in kfold_split(xtr[i].shape[0], splits):
        # reset to initial weights
        # model.set_weights(w_save)
        # fit with next kfold data
        model.fit(xtr[i][train], ytr[i][train],
                  batch_size=64, epochs=50, verbose=0)

        loss, accuracy = model.evaluate(xtr[i][val], ytr[i][val], verbose=0)
        avgacc[i] += accuracy

    avgacc[i] /= splits
    print("sub: {}  acc: {}".format(i + 1 if i + 1 < 10 else i + 2, avgacc[i]))
print("model: {}   avgacc: {}".format(j, sum(avgacc) / n_subs))
