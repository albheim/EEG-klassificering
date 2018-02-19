from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import Lambda, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, SimpleRNN, RNN, LSTM, GRU

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K


def lstm_lstm(input_shape, first_size=32, second_size=16, dropout_p=0.5, return_sequences=True):
    model = Sequential()
    model.add(LSTM(first_size, input_shape=input_shape, return_sequences=return_sequences))
    model.add(Dropout(dropout_p))
    model.add(LSTM(second_size))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def lstm_dense(input_shape, first_size=32, second_size=16, dropout_p=0.5):
    model = Sequential()
    model.add(LSTM(first_size, input_shape=input_shape))
    model.add(Dropout(dropout_p))
    model.add(Dense(second_size))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model
