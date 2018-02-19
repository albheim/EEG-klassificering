import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import TimeDistributed
from keras.layers import SimpleRNN, RNN, LSTM, GRU

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import models
import data
import util


x, y = data.load_single(cut=True)

splits = 5
n_subs = len(x)
n_models = 10


avgacc = 0

for i in range(n_subs):
    print("################ SUB {} ####################".format(i + 1 if i + 1 < 10 else i + 2))
    n = x[i].shape[0]
    acc = 0
    for tr, val in util.kfold(n, splits):

        pred = np.zeros((len(val), 3))

        for j in range(n_models):

            model = models.lstm_dense(x[0][0].shape,
                                      70, 20, 0.1)

            # fit with next kfold data
            model.fit(x[i][tr], y[i][tr],
                      batch_size=64, epochs=50, verbose=0)

            pred += model.predict(x[i][val], verbose=0)

        pred /= n_models
        acc += np.mean(np.equal(np.argmax(pred, axis=-1),
                                np.argmax(y[i][val], axis=-1)))

    acc /= splits
    avgacc += acc

    print("subject {}, avg accuracy {} over {} splits".format(i + 1 if i + 1 < 10 else i + 2, avgacc, splits))

avgacc /= n_subs
print("avg accuracy over all subjects {}".format(avgacc))


