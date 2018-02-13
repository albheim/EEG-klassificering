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
        for j in range(trials.shape[0]):
            trials[j] = trials[j][:, 768:1536]
        labels = np.zeros((trials.shape[0], 3))
        labels[:, i] = 1
        if x is None:
            x = trials
            y = labels
        else:
            x = np.concatenate((x, trials), axis=0)
            y = np.concatenate((y, labels), axis=0)

n = x.shape[0]
tr = int(1 * n)  # Currently use all data in training set

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


def get_random_setting():
    second_layer = np.random.choice([True, False])
    second_layer_type = np.random.choice([LSTM, Dense])
    second_layer_nodes = np.random.randint(10, 100)
    second_layer_dropout = np.random.ranf() * 0.5 + 0.25

    first_layer_nodes = np.random.randint(10, 100)
    first_layer_dropout = np.random.ranf() * 0.5 + 0.25
    if second_layer:
        first_layer_return_seq = np.random.choice([True, False])
    else:
        first_layer_return_seq = False

    output = np.random.choice([Dense])

    epochs = np.random.randint(10, 100)

    return {"first_layer_nodes": first_layer_nodes,
            "first_layer_dropout": first_layer_dropout,
            "first_layer_return_seq": first_layer_return_seq,
            "second_layer": second_layer,
            "second_layer_type": second_layer_type,
            "second_layer_nodes": second_layer_nodes,
            "second_layer_dropout": second_layer_dropout,
            "output_type": output,
            "epochs": epochs}



splits = 10
n_models = 1000
avgacc = [0 for i in range(n_models)]
models = [None for i in range(n_models)]

kfold = [i for i in kfold_split(xtr.shape[0], splits)]

for i in range(n_models):
    models[i] = get_random_setting()
    model = Sequential()

    # returns a sequence of vectors of dimension 32
    model.add(LSTM(models[i]["first_layer_nodes"],
                   input_shape=xtr[0].shape,
                   return_sequences=models[i]["first_layer_return_seq"]))
    model.add(Dropout(models[i]["first_layer_dropout"]))

    if models[i]["second_layer"]:
        model.add(models[i]["second_layer_type"](models[i]["second_layer_nodes"], activation='tanh'))
        #model.add(Dropout(models[i]["second_layer_dropout"]))

    model.add(models[i]["output_type"](3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    w_save = model.get_weights()

    model.summary()

    for train, val in kfold:
        # reset to initial weights
        model.set_weights(w_save)
        # fit with next kfold data
        model.fit(xtr[train], ytr[train],
                  batch_size=64, epochs=models[i]["epochs"], verbose=0)

        loss, accuracy = model.evaluate(xtr[val], ytr[val], verbose=0)
        avgacc[i] += accuracy

    avgacc[i] /= splits
    print("acc: {}   {}".format(avgacc[i], models[i]))

sort = np.argsort(avgacc)
avgacc = avgacc[sort]
models = models[sort]

print("\n\n")
print("\n".join(["acc: {}   {}".format(avgacc[i], models[i]) for i in range(n_models)]))
