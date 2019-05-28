# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = [arr.reshape(-1, 28, 28, 1) / 255. for arr in (x_train, x_test)]
y_train, y_test = [arr.reshape(-1, 1) for arr in (y_train, y_test)]


class MyLayer(keras.layers.Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)  # activation is not default in layer
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print('build')
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        print('call')
        return self.activation(tf.matmul(inputs, self.kernel))

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


x = keras.layers.Input(shape=(28, 28, 1))
y = keras.layers.Conv2D(32, (3, 3))(x)
y = keras.layers.Conv2D(32, (3, 3))(y)
y = keras.layers.MaxPool2D()(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dense(10, activation='softmax')(y)

model = keras.models.Model(inputs=[x], outputs=[y])
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test))
model.summary()

# similar to

x2 = keras.layers.Input(shape=(28, 28, 1))
y2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(x2)
y2 = keras.layers.Conv2D(32, (3, 3))(y2)
y2 = keras.layers.MaxPool2D()(y2)
y2 = keras.layers.Flatten()(y2)
y2 = MyLayer(10, activation='softmax')(y2)

model2 = keras.models.Model(inputs=[x2], outputs=[y2])
model2.compile(loss='sparse_categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
model2.fit(x_train, y_train, validation_data=(x_test, y_test))
model2.summary()

# it works. Just be aware of tf.Tensorshape.
x3 = keras.layers.Input(shape=(28, 28, 1))
y3 = keras.layers.Flatten()(x3)
y3 = MyLayer(100, activation='selu')(y3)
y3 = MyLayer(100, activation='selu')(y3)
y3 = MyLayer(100, activation='selu')(y3)
y3 = MyLayer(10, activation='softmax')(y3)

model3 = keras.models.Model(inputs=[x3], outputs=[y3])
model3.compile(loss='sparse_categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
model3.fit(x_train, y_train, validation_data=(x_test, y_test))
model3.summary()

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 28, 28, 1)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
my_layer_1 (MyLayer)         (None, 100)               78400     
_________________________________________________________________
my_layer_2 (MyLayer)         (None, 100)               10000     
_________________________________________________________________
my_layer_3 (MyLayer)         (None, 100)               10000     
_________________________________________________________________
my_layer_4 (MyLayer)         (None, 10)                1000      
=================================================================
Total params: 99,400
Trainable params: 99,400
Non-trainable params: 0
_________________________________________________________________

"""
