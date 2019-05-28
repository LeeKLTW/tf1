# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, output_dim, activation=None, use_bias=True,
                 kernel_initializer='lecun_normal', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        super().__init__(**kwargs)

        self.hidden = [keras.layers.Dense(output_dim, activation=activation, kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer) for _ in range(self.n_layers)]

    # def build(self, input_shape):
    #     pass

    # def compute_output_shape(self, input_shape):
    #     pass

    def call(self, inputs):
        z = inputs
        for layer in self.hidden:  # skip residual
            z = layer(z)
        z = z + inputs
        return z

    def get_config(self):
        pass

    @classmethod
    def from_config(cls, **config):
        return cls(**config)


class ResidualRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.flatten = keras.layers.Flatten()
        self.hidden1 = keras.layers.Dense(output_dim)
        self.residual_block1 = ResidualBlock(2, 512)
        self.residual_block2 = ResidualBlock(2, 512)
        self.out = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        y = self.flatten(inputs)
        y = self.hidden1(y)
        for _ in range(5):
            y = self.residual_block1(y)
        y = self.residual_block2(y)
        y = self.out(y)
        return y


def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = [arr.reshape(-1, 28, 28, 1) / 255. for arr in (x_train, x_test)]
    y_train, y_test = [arr.reshape(-1, 1) for arr in (y_train, y_test)]

    model = ResidualRegressor(512)
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test))
    model.summary()
