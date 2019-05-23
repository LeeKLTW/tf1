# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class ScaledDorProduct(keras.layers.Layer):
    def __init__(self, return_attention=False, history_only=False, **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        super(ScaledBorProduct, self).__init__(**kwargs)

    def get_config(self):
        config = {'return_attention': self.return_attention, 'history_only': self.history_only}
        base_config = super(ScaledBorProduct, self).get_config()
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
        if isinstance(mask,list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
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
