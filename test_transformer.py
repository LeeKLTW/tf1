# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from transformer import PositionalEncoding, MultiHeadAttention, AddNorm, PointwiseFFWD, EncoderBlock, Encoder
from transformer import MAX_LEN, MAX_WORD


def get_word(index):
    global word_index
    word_index = keras.datasets.reuters.get_word_index()
    for (k, v) in word_index.items():
        if v == index:
            return str(k)
    return ''


(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()
test_size = -1000  # -1 for all
(x_train, y_train), (x_test, y_test) = (x_train[:test_size], y_train[:test_size]), (
    x_test[:int(test_size / 10)], y_test[:int(test_size / 10)])

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

x = keras.layers.Input((MAX_LEN,))
y = keras.layers.Embedding(MAX_WORD * 10, 64)(x)
y = PositionalEncoding()(K.concatenate([y, y, y]))
y = K.reshape(y,(-1,MAX_LEN,))
y = MultiHeadAttention(8, 64)(y)
y = EncoderBlock()(y)  # (None , max_len, n_head, dim_k)
y = keras.layers.Flatten()(y)
y = keras.layers.Dense(46, activation='softmax')(y)

model = keras.Model(inputs=[x], outputs=[y])
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=tf.train.AdamOptimizer(),
              metrics=[keras.metrics.sparse_categorical_accuracy])

model.fit(x_train, y_train, epochs=1)
model.summary()
model.evaluate(x_test, y_test)
