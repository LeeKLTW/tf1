# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

input_shape = x_train.shape[-1:]

inp = keras.layers.Input(shape=input_shape)
y = keras.layers.Dense(30, activation='selu', kernel_initializer='lecun_normal')(inp)
y = keras.layers.Dense(30, activation='selu', kernel_initializer='lecun_normal')(y)
y = keras.layers.Concatenate()([y, inp])
y = keras.layers.Dense(30)(y)
# y = keras.layers.Dense(30,kernel_regularizer=keras.regularizers.l2(0.01))(y)
y = keras.layers.Dropout(0.5)(y)
y = keras.layers.Dense(30)(y)
# y = keras.layers.Dense(30,kernel_regularizer=keras.regularizers.l2(0.01))(y)
y = keras.layers.Dropout(0.5)(y)
y = keras.layers.Dense(30)(y)
# y = keras.layers.Dense(30,kernel_regularizer=keras.regularizers.l2(0.01))(y)
y = keras.layers.Dropout(0.5)(y)
y = keras.layers.Dense(1)(y)

# inp = keras.layers.Input(shape=input_shape)
# y = keras.layers.Dense(30,activation='relu')(inp)
# y = keras.layers.Dense(30,activation='relu')(y)
# y = keras.layers.Concatenate()([y,inp])
# y = keras.layers.Dense(1)(y)

# optimizer = keras.optimizers.Adam()
optimizer = keras.optimizers.Nadam(1e-4)

model = keras.Model(inputs=[inp], outputs=[y])
model.compile(loss='mse', optimizer=optimizer, metrics=[keras.metrics.mean_squared_error])
model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), workers=4, use_multiprocessing=True)
model.summary()

model_clone = keras.models.clone_model(model)
model_clone.summary()
model.evaluate(x_test, y_test)
model_clone.set_weights(model.get_weights())
for layer in model_clone.layers[:-1]:
    layer.trainable = False
model_clone.compile(loss='mse', optimizer=optimizer, metrics=[keras.metrics.mean_squared_error])

model_clone.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), workers=4, use_multiprocessing=True)



def mse(y_true, y_pred):
    error = tf.sqrt(tf.pow(y_true - y_pred, 2))
    error = tf.reduce_sum(error) / tf.cast(tf.shape(y_true)[0], 'float32')
    return error


def mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    error = tf.reduce_sum(error) / tf.cast(tf.shape(y_true)[0], 'float32')
    return error


def huber_fn(y_true, y_pred):
    error = y_true - y_pred


# class HuberLoss(keras.losses.Loss): # this should be available in tf2.0
class HuberLoss:
    # super(HuberLoss, self).__init__(**kwargs)
    def __init__(self, delta=5, **kwargs):
        self.delta = delta

    def __call__(self, y_true, y_pred):
        error = y_true - y_pred
        squred_loss = tf.pow(error, 2) / 2
        abs_loss = self.delta * (tf.abs(error) - 0.5 * self.delta)
        is_smaller_than_delta = tf.abs(error) <= self.delta  # boolean
        tf.where(is_smaller_than_delta, abs_loss, squred_loss)

    def get_config(self):
        # base_config = super(HuberLoss,self).get_config()
        return {'delta': self.delta}


def my_softplus(z):
    return tf.math.log(1 + tf.math.exp(z))


def my_glorot_normal_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape=shape, mean=0.0, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.math.abs(weights))


def my_relu(z):
    return tf.where(tf.greater(z, 0), z, tf.zeros_like(z))



class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self):
        pass

    def __call__(self, weights):
        return tf.reduce_sum(tf.math.abs(weights))

    def get_config(self):
        return

class MyDense(keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        super(MyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1:].as_list()+[self.output_dim]))
        self.kernel = self.add_weight(name='kernel', shape=shape, initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias',shape=shape[-1], initializer='zeros',trainable=True)
        super(MyDense, self).build(input_shape)

    def call(self, inputs): #dont use __call__, use call, otherwise it will not call build
        z = tf.matmul(inputs, self.kernel)
        z = self.activation(z)
        return z

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        config = super(MyDense, self).get_config()
        config = {**config, 'output_dim':self.output_dim, 'activation':keras.activations.serialize(self.activation)}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = [arr.reshape(-1, 28, 28, 1) / 255. for arr in (x_train, x_test)]
    y_train, y_test = [arr.reshape(-1, 1) for arr in (y_train, y_test)]

    x3 = keras.layers.Input(shape=(28, 28, 1))
    y3 = keras.layers.Flatten()(x3)
    y3 = MyDense(100, activation='selu')(y3)
    y3 = MyDense(100, activation='selu')(y3)
    y3 = MyDense(100, activation='selu')(y3)
    y3 = MyDense(10, activation='softmax')(y3)

    model3 = keras.models.Model(inputs=[x3], outputs=[y3])
    model3.compile(loss='sparse_categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
    model3.fit(x_train, y_train, validation_data=(x_test, y_test))
    model3.summary()

