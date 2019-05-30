# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
tf.enable_eager_execution()

class PositionnalEncoding(keras.layers.Layer):
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 input_dim,
                 output_dim,
                 mode=MODE_EXPAND,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        """
        :param input_dim: The maximum absolute value of positions.
        :param output_dim: The embedding dimension.
        :param embeddings_initializer:
        :param embeddings_regularizer:
        :param activity_regularizer:
        :param embeddings_constraint:
        :param mask_zero: The index that represents padding. Only works in `append` mode.
        :param kwargs:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero is not False

        self.embeddings = None
        super().__init__(**kwargs)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'mode': self.mode,
                  'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            self.embeddings = self.add_weight(
                shape=(self.input_dim * 2 + 1, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if self.mode == self.MODE_EXPAND:
            if self.mask_zero:
                output_mask = K.not_equal(inputs, self.mask_zero)
            else:
                output_mask = None
        else:
            output_mask = mask
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, **kwargs):
        if self.mode == self.MODE_EXPAND:
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            return K.gather(
                self.embeddings,
                K.minimum(K.maximum(inputs, -self.input_dim), self.input_dim) + self.input_dim,
            )
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
        pos_embeddings = K.tile(
            K.expand_dims(self.embeddings[:seq_len, :self.output_dim], axis=0),
            [batch_size, 1, 1],
        )
        if self.mode == self.MODE_ADD:
            return inputs + pos_embeddings
        return K.concatenate([inputs, pos_embeddings], axis=-1)


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, n_head=8, dim_k=64, activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_model = int(n_head) * int(dim_k)
        self.output_dim = tf.Dimension(64)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.supports_masking = True  # use keras built-in mask

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
            if inputs.shape.ndims == 4:  # (batch_size, max_len, dim_k)
                q, k, v = inputs[0], inputs[1], inputs[2]
            elif inputs.shape.ndims == 3:  # (batch_size, max_len) self-attention
                q, k, v = inputs,inputs,inputs
        elif isinstance(inputs, list) or isinstance(inputs, tuple):  # [q,k,v] or (q,k,v)
            if len(inputs) == 3:
                q, k, v = inputs
        else:
            raise ValueError('Input [q,k,v]')

        q = K.concatenate([q] * self.n_head, axis=-1)
        k = K.concatenate([k] * self.n_head, axis=-1)
        v = K.concatenate([v] * self.n_head, axis=-1)

        q = self.WQbQ(q)  # [batch_size, maxlen, dim_model]
        k = self.WKbK(k)  # [batch_size, maxlen, dim_model]
        v = self.WVbV(v)  # [batch_size, maxlen, dim_model]
        y = K.batch_dot(q, k, axes=[-1, -1])  # [batch_size, d_model,d_model]
        scale = self.dim_k**(1/2)
        y = keras.layers.Lambda(lambda y:y/scale)(y)
        y = K.softmax(y)  # [batch_size, d_model, d_model]
        y = K.batch_dot(v, y, axes=[2, -1])  # [batch_size, maxlen, dim_model]
        return y

        pass

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


# From Long
class AddNorm(keras.layers.Layer):
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class PointwiseFFWD(keras.layers.Layer):
    def __init__(self, d_model=512, d_ff=2048, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1b1 = keras.layers.Conv1D(d_ff, kernel_size=1)
        self.w2b2 = keras.layers.Conv1D(d_model, kernel_size=1)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        y = self.w1b1(inputs)
        y = self.w2b2(y)
        return y


class EncoderBlock(keras.layers.Layer):
    def __init__(self, n_head=8, dim_k=64, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_model = int(n_head) * int(dim_k)

        self.mha = MultiHeadAttention(n_head=n_head,dim_k=dim_k)
        self.add_norm = AddNorm()
        self.ffwd = PointwiseFFWD()
        self.dropout = keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        y = self.dropout(inputs)
        x = y
        y = self.mha(y)
        y = self.dropout(y)
        y = self.add_norm([x, y])

        x = y
        y = self.ffwd(y)
        y = self.dropout(y)
        y = self.add_norm([x, y])

        return y


MAX_WORD = 1000
MAX_LEN = 10


class Encoder(keras.Model):
    def __init__(self, output_dim=64, n_blocks=6, n_category=46, **kwargs):
        super().__init__(**kwargs)
        self.time_dist = keras.layers.TimeDistributed(keras.layers.Dense(1))
        self.embedding = keras.layers.Embedding(MAX_WORD, output_dim)
        self.posenc = PositionnalEncoding(input_dim=MAX_LEN, output_dim=output_dim)
        self.encoder_blocks = [EncoderBlock() for _ in range(n_blocks)]
        self.output_dense = keras.layers.Dense(n_category, activation='softmax')

    def call(self, inputs):
        y = self.time_dist(inputs)
        y = self.embedding(y)
        y = self.posenc(y)

        for block in self.encoder_blocks:
            y = block(y)

        y = self.output_dense(y)
        return y

model = Encoder()
optimizer = tf.train.AdamOptimizer()
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=optimizer,
              metrics=[keras.metrics.sparse_categorical_accuracy])
(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()
test_run = True
if test_run:
    (x_train, y_train), (x_test, y_test) = (x_train[:100], y_train[:100]), (x_test[:10], y_test[:10])
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)
model.fit(x_train, y_train, validation_data=(x_test, y_test))
model.summary()
model.evaluate(x_test,y_test)

# todo fix line 363:  assert len(input_shape) >= 3