# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras


class Attention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.scale = output_dim ** (1 / 2)
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape([self.output_dim])
        self.WQ = self.add_weight(name='WQ', shape=shape.as_list(), initializer='glorot_normal', trainable=True)
        self.WK = self.add_weight(name='WQ', shape=shape.as_list(), initializer='glorot_normal', trainable=True)
        self.WV = self.add_weight(name='WQ', shape=shape.as_list(), initializer='glorot_normal', trainable=True)
        # use as_list() to fix TypeError: unsupported operand type(s) for /=: 'float' and 'Dimension'
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape([self.output_dim, self.output_dim])
        return shape

    def call(self, inputs):
        "inputs:[q,k,v]"
        q = tf.matmul(inputs[0], self.WQ)
        k = tf.matmul(inputs[1], self.WK)
        v = tf.matmul(inputs[2], self.WV)

        # todo: fix dtype to float32, & check neccessary for transpose.
        y = tf.matmul(q, k)
        y = keras.layers.Softmax()(y)

        y = tf.matmul(y, v)
        return y

    def get_config(self):
        pass

    @classmethod
    def from_config(cls, **config):
        return cls(**config)

# import numpy as np
# x_train = np.random.random((1,10,64))
#
# att = Attention(64)
# att([x_train,x_train,x_train])
# att.output_dim