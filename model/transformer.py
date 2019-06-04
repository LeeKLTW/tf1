# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np

# tf.enable_eager_execution()
MAX_WORD = 10000
MAX_LEN = 200


class PositionalEncoding(keras.layers.Layer):
    """3.5 Positional encoding."""

    def __init__(self, dim_k=64, n_head=8, maxlen=MAX_LEN, masking=False, **kwargs):
        self.output_dim = self.dim_model = dim_k * n_head  # dim_model
        self.maxlen = maxlen
        self.supports_masking = masking
        super().__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, maxlen=MAX_LEN, mode='add'):
        """inputs shape must be [None,maxlen,d_model] """
        MODE_LIST = ['add']

        pos_enc = K.variable(
            [[pos / (10000 ** (i / self.dim_model)) for i in range(self.dim_model)] for pos in range(maxlen)])
        tf.assign(pos_enc[0::2, :], tf.sin(pos_enc[0::2, :]))
        tf.assign(pos_enc[1::2, :], tf.cos(pos_enc[1::2, :]))

        if mode.lower() == 'add':
            y = inputs + pos_enc

        else:
            raise ValueError(f'argument of mode must be in {MODE_LIST[:]}, but recieve {mode}')

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, n_head=8, dim_k=64, masking_value=0.0, activation='relu', use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, reshape_to_head=False, **kwargs):
        """

        Args:
            n_head:
            dim_k:
            masking_value: set to -1e8, to use masking
            activation:
            use_bias:
            kernel_initializer:
            bias_initializer:
            kernel_regularizer:
            bias_regularizer:
            kernel_constraint:
            bias_constraint:
            reshape_to_head:
            **kwargs:
        """
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_model = int(n_head) * int(dim_k)
        self.output_dim = tf.Dimension(64)
        self.masking_value = masking_value

        self.activation = keras.activations.get(activation)

        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.supports_masking = True  # use keras built-in mask
        self.reshape_to_head = reshape_to_head

        self.WQbQ = keras.layers.Dense(self.dim_model, kernel_initializer=self.kernel_initializer,
                                       bias_initializer=self.bias_initializer, trainable=True)

        self.WKbK = keras.layers.Dense(self.dim_model, kernel_initializer=self.kernel_initializer,
                                       bias_initializer=self.bias_initializer, trainable=True)

        self.WVbV = keras.layers.Dense(self.dim_model, kernel_initializer=self.kernel_initializer,
                                       bias_initializer=self.bias_initializer, trainable=True)

        super().__init__(**kwargs)

    # def build(self, input_shapes):
    #     if len(input_shapes[0]) != len(input_shapes[1]) != len(input_shapes[2]):
    #         raise ValueError('Q, K, V Rank are not equal')  # (batch_size, max_len, dim_k)

    def call(self, inputs):
        if isinstance(inputs, tf.Tensor):
            if inputs.shape.ndims == 4:  # (batch_size, max_len, dim_model)
                q, k, v = inputs[0], inputs[1], inputs[2]
            elif inputs.shape.ndims == 3:  # (batch_size, max_len) self-attention
                q, k, v = inputs, inputs, inputs
        elif isinstance(inputs, list) or isinstance(inputs, tuple):  # [q,k,v] or (q,k,v)
            if len(inputs) == 3:
                q, k, v = inputs
            else:
                raise ValueError(f'Length of list is not correct{len(inputs)}')
        else:
            raise ValueError('Input [q,k,v]')

        # linear W
        q = self.WQbQ(q)  # [batch_size, maxlen, dim_model]
        k = self.WKbK(k)  # [batch_size, maxlen, dim_model]
        v = self.WVbV(v)  # [batch_size, maxlen, dim_model]

        # matmul
        y = K.batch_dot(q, k, axes=[-1, -1])  # [batch_size, d_model,d_model]

        # scale
        scale = self.dim_k ** (1 / 2)
        y = keras.layers.Lambda(lambda y: y / scale)(y)  # [batch_size, d_model, d_model]

        # mask(opt)
        if self.masking_value != 0.:
            y = keras.layers.Masking(self.masking_value)(y)

        # softmax

        y = K.reshape(y, (-1, self.dim_model, self.n_head, self.dim_k))  # [batch_size, dim_model, n_head, dim_k]
        y = K.softmax(y, axis=-1)  # [batch_size, dim_model, dim_model]
        y = K.reshape(y, (-1, self.dim_model, self.dim_model))  # [batch_size, dim_model, dim_model]

        # mat_mul
        y = K.batch_dot(v, y, axes=[2, 3])  # [batch_size, maxlen, dim_model]

        if self.reshape_to_head:
            y = K.reshape(y, (-1, q.shape[1], self.n_head, self.dim_k))

        return y

    def get_config(self):
        config = {'head_num': self.head_num,
                  'activation': keras.activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
                  'bias_initializer': keras.initializers.serialize(self.bias_initializer),
                  'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint),
                  'history_only': self.history_only
                  }

        base_config = super(MultiHeadAttention, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        return shape


class AddNorm(keras.layers.Layer):
    def __init__(self, epsilon=K.epsilon(),
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):
        """Layer normalization layer
        """
        super().__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            if isinstance(input_shape[0], tf.TensorShape):
                assert input_shape[0].as_list() == input_shape[1].as_list(), 'shape must be equal'
                shape = input_shape[0].as_list()[1:]
            else:
                raise TypeError(f'Unrecognized type {type(input_shape[0])}')
        elif isinstance(input_shape, tf.TensorShape):
            shape = input_shape.as_list()[1:]
        else:
            raise TypeError(f'Unrecognized type {type(input_shape)}')
        self.gamma = self.add_weight(
            shape=shape,
            initializer=self.gamma_initializer,
            name='gamma')
        self.beta = self.add_weight(
            shape=shape,
            initializer=self.beta_initializer,
            name='beta')
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise TypeError('Need to be list for residual.')

        # y = keras.layers.Add()([inputs[0], inputs[1]])
        y = inputs[0]+inputs[1] # fix [None, None] above

        mean = K.mean(y, axis=-1, keepdims=True)
        std = K.std(y, axis=-1, keepdims=True)
        y = self.gamma * (y - mean) / (std + self.epsilon) * self.beta
        return y


class PointwiseFFWD(keras.layers.Layer):
    def __init__(self, d_model=512, d_ff=2048, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1b1 = keras.layers.Conv1D(d_ff, kernel_size=1, padding='same')
        self.w2b2 = keras.layers.Conv1D(d_model, kernel_size=1, padding='same')

    def build(self, input_shape):
        pass

    def call(self, inputs):
        y = self.w1b1(inputs)
        y = self.w2b2(y)
        return y


# todo
class EncoderBlock(keras.layers.Layer):
    def __init__(self, n_head=8, dim_k=64, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_model = int(n_head) * int(dim_k)

        self.mha = MultiHeadAttention(n_head=n_head, dim_k=dim_k)
        self.add_norm = AddNorm()
        self.ffwd = PointwiseFFWD()
        self.dropout = keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        x = y = self.dropout(inputs)
        y = self.mha([y, y, y])
        y = self.dropout(y)

        x = K.expand_dims(x, -2)
        x = K.concatenate([x] * self.n_head, axis=-2)

        x = y = self.add_norm([x, y])

        y = self.ffwd(y)
        y = self.dropout(y)

        y = self.add_norm([x[1:], y[1:]])

        return y


class Encoder(keras.Model):
    def __init__(self, output_dim=64, n_blocks=6, n_category=46, **kwargs):
        super().__init__(**kwargs)
        self.embedding = keras.layers.Embedding(MAX_WORD, output_dim)
        self.posenc = PositionalEncoding(input_dim=MAX_LEN, output_dim=output_dim)
        self.encoder_blocks = [EncoderBlock() for _ in range(n_blocks)]
        self.output_dense = keras.layers.Dense(n_category, activation='softmax')

    def call(self, inputs):
        y = inputs
        y = self.embedding(y)
        # y = self.posenc(y)

        for block in self.encoder_blocks:
            y = block(y)

        y = self.output_dense(y)
        return y
