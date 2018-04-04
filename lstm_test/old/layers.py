import numpy as np

from keras import backend as K
from keras.engine import Layer

class GaussianNoiseAlways(Layer):

    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseAlways, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs):
        return inputs + K.random_normal(shape=K.shape(inputs),
                                        mean=0.,
                                        stddev=self.stddev)

    def compute_output_shape(self, input_shape):
        return input_shape
