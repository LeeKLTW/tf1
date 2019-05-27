# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf

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

import tensorflow as tf

t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype='float32')
t @ tf.transpose(t)

tf.constant(2.) + tf.constant(2., dtype='float32')

var = tf.Variable([1, 2], dtype='float32')
var = var.assign(var + 2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(var))


def mse(y_true, y_pred):
    error = tf.sqrt(tf.pow(y_true - y_pred, 2))
    error = tf.reduce_sum(error) / tf.cast(tf.shape(y_true)[0], 'float32')
    return error


def mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    error = tf.reduce_sum(error) / tf.cast(tf.shape(y_true)[0], 'float32')
    return error


yt = tf.constant([1, 3], dtype='float32')
yp = tf.constant([1, 3], dtype='float32')

with tf.Session() as sess:
    print(sess.run(mse(yt, yp)))



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
        return {'delta':self.delta}