# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class Attention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.scale = output_dim ** (1 / 2)
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape([self.output_dim, self.output_dim])
        self.WQ = self.add_weight(name='WQ', shape=shape.as_list(), initializer='glorot_normal', trainable=True,
                                  dtype='float32')
        self.WK = self.add_weight(name='WQ', shape=shape.as_list(), initializer='glorot_normal', trainable=True,
                                  dtype='float32')
        self.WV = self.add_weight(name='WQ', shape=shape.as_list(), initializer='glorot_normal', trainable=True,
                                  dtype='float32')
        # use as_list() to fix TypeError: unsupported operand type(s) for /=: 'float' and 'Dimension'
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape([self.output_dim, self.output_dim])
        return shape

    def call(self, inputs):
        "inputs:[q,k,v]"
        q = tf.matmul(inputs[0], self.WQ) #
        k = tf.matmul(inputs[1], self.WK) #
        v = tf.matmul(inputs[2], self.WV) #

        y = tf.matmul(q, k,transpose_b=True)
        y = keras.layers.Softmax()(y)

        y = tf.matmul(y, v)
        return y

    def get_config(self):
        pass

    @classmethod
    def from_config(cls, **config):
        return cls(**config)


import numpy as np

x_train = np.random.random((10, 64)).astype('float32')

att = Attention(64)
y= att([x_train, x_train, x_train])
K.eval(y)
y.shape
#
#
#
# x = keras.layers.Input(shape=(10,64,))
# y = Attention(64)(x)
#
# model = keras.Model(inputs=[x],outputs=[y])
# model.compile()
