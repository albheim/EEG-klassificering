import sys
from datetime import datetime

import numpy as np
from scipy import io

import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, Input, GaussianNoise, BatchNormalization
from keras.layers import Lambda
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import ELU, Activation, Flatten

from keras import backend as K
from keras import regularizers as rg

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import data
import util


x, y = data.load_transform([5], 'cwt')

splits = 5

# channels = [4, 23]
# for i in range(n_subs):
#     x[i] = x[i][:, :, channels]
#     xt[i] = xt[i][:, :, channels]


m_in = Input(shape=x[0][0].shape)

m_t = Conv2D(4, (4, 8), padding='same', kernel_regularizer=rg.l1(0.01))(m_in)
#m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling2D((2, 4))(m_t)
m_t = Dropout(0.2)(m_t)

m_t = Conv2D(8, (4, 8), padding='same', kernel_regularizer=rg.l1(0.01))(m_t)
#m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling2D((2, 4))(m_t)
m_t = Dropout(0.3)(m_t)

m_t = Conv2D(16, (4, 8), padding='same', kernel_regularizer=rg.l1(0.01))(m_t)
#m_t = BatchNormalization()(m_t)
m_t = ELU()(m_t)
m_t = AveragePooling2D((2, 2))(m_t)
m_t = Dropout(0.4)(m_t)

m_t = Flatten()(m_t)
m_t = Dense(15, kernel_regularizer=rg.l1(0.01))(m_t)
#m_t = BatchNormalization()(m_t)
m_t = Activation('tanh')(m_t)
m_out = Dense(3, activation='softmax')(m_t)

model = Model(inputs=m_in, outputs=m_out)

m_save = model.get_config()
model.summary()

acc = 0
for tr, val in util.kfold(len(x[0]), splits, shuffle=True):

    model = Model.from_config(m_save)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # fit with next kfold data
    h = model.fit(x[0][tr], y[0][tr],
                  validation_data=(x[0][val], y[0][val]),
                  batch_size=16, epochs=50, verbose=1)

    _, a = model.evaluate(x[0][val], y[0][val], verbose=0)
    acc += a

acc /= splits

print("avg accuracy {} over {} splits".format(acc, splits))
