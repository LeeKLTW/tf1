# -*- coding: utf-8 -*-
"""

"""
import numpy as np

import tensorflow as tf

# tf.enable_eager_execution()
# tf.executing_eagerly()
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# x_train = ['大部分的理賠案都是直接理賠成功或是經由財團法人金融評議中心調處後達成共識，只有少數的爭議案件走到訴訟程序',
#            'OpenBSD以對開放原始碼的堅持、高品質的檔案、堅定的軟體受權條款和專注於系統安全及程式碼品質而聞名']
# y_train = [1, 0]
#
# np.random.seed(42)
# # NUM_WORDS = max([len(sent) for sent in x_train])  # 2376
# NUM_WORDS = 3000
# MAX_LEN = 60
#
# tokenizer = Tokenizer(num_words=NUM_WORDS, char_level=True, oov_token=1)
# tokenizer.fit_on_texts(''.join(x_train))
# x_train = tokenizer.texts_to_sequences(x_train)
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
#
# model = keras.Sequential()
# model.add(keras.layers.Embedding(input_dim=MAX_LEN + 100, output_dim=64, input_shape=(MAX_LEN,)))
# model.add(keras.layers.Dense(10, activation='relu'))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10)
# model.summary()


# def idx2word(idx):
#     return [w for w in word_index if word_index[w] == idx][0]


# def _tokenize():
#     x_example_idx = np.random.choice(range(len(x_train)), 1)[0]
#     x_example = ' '.join([idx2word(idx) for idx in x_train[x_example_idx]])
#     print(x_example)
#     NUM_WORDS = max([len(sent) for sent in x_train])  # 2376
#     tokenizer = Tokenizer(num_words=NUM_WORDS)
#     tokenizer.fit_on_texts([x_example])
#     print(tokenizer.texts_to_sequences([x_example]))
#
#
# def _pad_x(x):
#     return pad_sequences(x, maxlen=NUM_WORDS)


class MultiheadAttention(keras.layers.Layer):
    def __init__(self, d_k, n_head, mask_idx=None, **kwargs):
        self.d_k = d_k
        self.n_head = n_head
        self.n_model = self.d_k * self.n_head
        self.mask_idx = mask_idx
        super(MultiheadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # inputs = [q,k,v]
        self.WQ = self.add_weight(name='WQ', shape=(input_shape[0][-1], self.n_model),
                                  trainable=True)  # shape=(d_model,d_model)
        self.WK = self.add_weight(name='WK', shape=(input_shape[1][-1], self.n_model),
                                  trainable=True)  # shape=(d_model,d_model)
        self.WV = self.add_weight(name='WV', shape=(input_shape[2][-1], self.n_model),
                                  trainable=True)  # shape=(d_model,d_model)

        self.WO = self.add_weight(name='WO', shape=(input_shape[2][-1], self.n_model),
                                  trainable=True)  # shape=(d_model,d_model)

        super(MultiheadAttention, self).build(input_shape)

    def mask(self, attn, mask_idx, used_padding='pre'):
        """attn.shape = (batch_size, max_len, n_head, d_k)
        add -infinity not multiplication, cause it might be 0, make it failed to mask
        """
        mask_unseen = -K.ones(shape=(K.shape(attn)[0], mask_idx, K.shape(attn)[2], K.shape(attn)[3])) * 1e10
        seen = K.zeros(shape=(K.shape(attn)[0], mask_idx - 1, K.shape(attn)[2], K.shape(attn)[3]))

        if used_padding.lower() == 'pre':  # pad_sequences default padding='pre'
            mask = K.concatenate([mask_unseen, seen], axis=1)

        elif used_padding.lower() == 'post':
            mask = K.concatenate([seen, mask_unseen], axis=1)
        else:
            raise ValueError(
                'Please Insert correct padding arg used in tensorflow.keras.preprocessing.sequence.pad_sequences')
        attn += mask

        return attn

    def __call__(self, inputs):
        """inputs = [q,k,v] or [q,k,v,mask_len]"""
        if len(inputs) == 3:
            q, k, v = inputs
        elif len(inputs) == 4:
            q, k, v, self.mask_idx = inputs

        q = K.batch_dot(q, self.WQ)  # shape =(batch_size,max_len ,d_model)
        q = K.reshape(q, (-1, K.shape(q)[1], self.n_head, self.d_k))  # shape = (batch_size, max_len, n_head, d_k)

        k = K.batch_dot(k, self.WK)  # shape =(batch_size,max_len ,d_model)
        k = K.reshape(k, (-1, K.shape(k)[1], self.n_head, self.d_k))  # shape = (batch_size, max_len, n_head, d_k)

        v = K.batch_dot(v, self.WV)  # shape =(batch_size,max_len ,d_model)
        v = K.reshape(v, (-1, K.shape(v)[1], self.n_head, self.d_k))  # shape = (batch_size, max_len, n_head, d_k)

        a = K.batch_dot(q, k, axes=[3, 3]) / self.d_k ** 0.5  # shape = (batch_size, max_len, n_head, d_k)
        a = self.mask(a, self.mask_idx)

        a = K.softmax(a)

        a = K.batch_dot(a, v, axes=[3, 3]) / self.d_k ** 0.5  # shape = (batch_size, max_len, n_head, d_k)

        # concat(heads)*WO  since we compute together, we dont have to concat,
        a = K.batch_dot(a, self.WO)  # shape = (batch_size, max_len, n_head, d_k)
        return a


    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormalization(keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape



def foo():
    q = np.random.random((1, 12, 512))
    mha = MultiHeadAttention()
    x = mha(q, q, q)
    ln = LayerNormalization()
    x = ln(x)
    x


x = np.arange(0,15).reshape(5,3)
x = keras.layers.Lambda()
tf.reshape(x,[1,12,512])
ffn = PositionwiseFFN()
ffn(tf.reshape(x,[1,12,512]))

class PositionwiseFFN:
    def __init__(self, d_model=512, d_ff=2048):
        self.w_1 = keras.layers.Conv1D(d_ff, 1, activation='relu')
        self.w_2 = keras.layers.Conv1D(d_model, 1)  # linear

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        return output


class EncoderLayer:
    def __init__(self, d_model, d_ff, n_head, dropout_rate=0.1):
        self.multi_head_attention = MultiHeadAttention(n_head, d_model, dropout_rate)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.addnorm1 = ADDNORM()
        self.pos_ffn = PositionwiseFFN(d_model, d_ff)
        self.addnorm2 = ADDNORM()

    def __call__(self, enc_inputs, mask):
        """enc_inputs: input embedding after Positional Encoding."""
        multihead, multiattention = self.multi_head_attention(enc_inputs, enc_inputs, enc_inputs, mask=mask)
        multihead = self.dropout(multihead)
        multihead_addnorm = self.addnorm1(enc_inputs, multihead)

        pos_ffn_output = self.pos_ffn(multihead_addnorm)
        pos_ffn_output = self.dropout(pos_ffn_output)
        pos_ffn_output_addnorm = self.addnorm2(multihead_addnorm, pos_ffn_output)

        return pos_ffn_output_addnorm, multiattention


class DecoderLayer:
    def __init__(self, d_model, d_ff, n_head, dropout_rate=0.1):
        self.masked_multi_head_attention = MultiHeadAttention(n_head, d_model, dropout_rate)
        self.addnorm1 = ADDNORM()

        self.multi_head_attention = MultiHeadAttention(n_head, d_model, dropout_rate)
        self.addnorm2 = ADDNORM()

        self.pos_ffn = PositionwiseFFN(d_model, d_ff)
        self.addnorm3 = ADDNORM()

    def __call__(self, dec_inputs, enc_outputs, dec_mask=None, enc_mask=None, dec_last_state=None):
        dec_last_state = dec_inputs if dec_last_state is None else dec_last_state
        m_multihead, dec_multiattention = self.masked_multi_head_attention(dec_inputs, dec_last_state, dec_last_state,
                                                                           mask=enc_mask)
        m_multihead_addnorm = self.addnorm1(dec_inputs, m_multihead)

        multihead, enc_multiattention = self.multi_head_attention(m_multihead_addnorm, enc_outputs, enc_outputs,
                                                                  mask=enc_mask)
        multihead_addnorm = self.addnorm2(m_multihead, multihead)

        pos_ffn_output = self.pos_ffn(multihead_addnorm)
        pos_ffn_output_addnorm = self.addnorm2(multihead_addnorm, pos_ffn_output)
        return pos_ffn_output_addnorm, dec_multiattention, enc_multiattention


class Encoder:
    def __init__(self, d_model=512, d_ff=2048, n_head=8, n_layers=6, dropout_rate=0.1):
        self.layers = [EncoderLayer(d_model, d_ff, n_head, dropout_rate) for _ in range(n_layers)]

    def __call__(self, inputs_embedding, input_sequence, mask, return_attentions=False):
        if return_attentions:
            attentions = []
        x = inputs_embedding
        for enc_layer in self.layers:
            x, att = enc_layer(x, mask=mask)
            if return_attentions:
                attentions.append(att)
        return (x, attentions) if return_attentions else x


def get_PAD_mask(q, k):  # todo check it latter
    ones = K.expand_dims(K.ones_like(q, dtype='float32'), -1)
    mask = K.cast(K.expand_dims(K.not_qual(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def get_sub_mask(s):  # todo check it latter
    len_s = K.shape(s)[1]
    mask = K.cumsum(K.eye(len_s), 1)
    return mask


class Deocoder:
    def __init__(self, d_model, d_ff, n_head=8, n_layers=6, dropout_rate=0.1):
        self.layers = [DecoderLayer(d_model, d_ff, n_head, dropout_rate) for _ in range(n_layers)]

    def __call__(self, outputs_embedding, input_seq, output_seq, encoder_output, return_attentions=False):
        if return_attentions:
            encoder_attentions = []
            decoder_attentions = []

        dec_pad_mask = keras.layers.Lambda(lambda x: get_PAD_mask(x, x))(output_seq)
        dec_sub_mask = keras.layers.Lambda(get_sub_mask)(output_seq)
        dec_mask = keras.layers.Lambda(lambda x: K.minimum(x[0], x[1]))([dec_pad_mask, dec_sub_mask])
        enc_mask = keras.layers.Lambda(lambda x: get_PAD_mask(x[0], x[1]))([output_seq, input_seq])

        x = outputs_embedding
        for dec_layer in self.layers:
            x, dec_multiattention, enc_multiattention = dec_layer(x, encoder_output, dec_mask, enc_mask)
            if return_attentions:
                decoder_attentions.append(dec_multiattention)
                encoder_attentions.append(enc_multiattention)
        return (x, decoder_attentions, encoder_attentions) if return_attentions else x


class Transformer:
    def __init__(self, maxlen, d_model=512, d_ff=2048, n_head=8, d_v=64, layers=2, dropout=0.1):
        self.input_embedding = None  # dot it here
        self.positional_encoding = None  # do it here
        self.output_embedding = None
        self.encoder = None
        self.decoder = None
        self.linear = None
        self.softmax = None


class Transformer(keras.Model):
    def __init__(self, in_dim=784, out_dim=10, d_model=512, n_layers=10, dp_rate=0.5):
        self.n_model = d_model
        self.n_layers = n_layers
        self.dp_rate = dp_rate
        self.out_dim = out_dim

        super(Transformer, self).__init__(name='Transformer')

        self.input_layer = keras.layers.Input(shape=(784,))
        for i in range(self.n_layers):
            exec(f"self.dense_{i + 1} = keras.layers.Dense(d_model,activation='relu',name='dense_{i + 1}')")
            if self.dp_rate > 0:
                exec(f"self.dp_{i + 1} = keras.layers.Dropout(dp_rate,name='dp_{i + 1}')")
