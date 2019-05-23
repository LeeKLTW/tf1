# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class ScaledDotProduct(keras.layers.Layer):
    def __init__(self, return_attention=False, history_only=False, **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        super(ScaledDotProduct, self).__init__(**kwargs)

    def get_config(self):
        config = {'return_attention': self.return_attention, 'history_only': self.history_only}
        base_config = super(ScaledDotProduct, self).get_config()
        config = list(base_config.items()) + list(config.items())
        return config

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape

        output_shape = query_shape[:-1] + value_shape[:-1]

        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]

        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def __call__(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]

        feature_dim = K.shape(query)[-1]
        dvd = K.sqrt(K.cast(feature_dim, K.floatx(),'float64')) # for sqrt

        a = K.batch_dot(query, key, axes=-1) / dvd
        a = K.exp(a - K.max(a, axis=-1, keepdims=True))  # not softmax directly, need to mask

        if self.history_only:
            # query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            # indicied = tf.tile()
            pass
        if mask is not None:
            a += K.cast(K.expand_dims(mask, axis=-2), K.floatx())

        a = a / (K.sum(a, axis=-1, keepdims=True) + K.epsilon())
        v = K.batch_dot(a, value, axes=2)
        if self.return_attention:  # residual use
            return [v, a]
        return v


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, n_head, activation='relu', use_bias=True):
        pass
