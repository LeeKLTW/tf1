# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from model.transformer import PositionalEncoding,PointwiseFFWD,MultiHeadAttention,AddNorm
from model.transformer import MAX_WORD,MAX_LEN

def test():

    def get_word(index):
        global word_index
        word_index = keras.datasets.reuters.get_word_index()
        for (k, v) in word_index.items():
            if v == index:
                return str(k)
        return ''

    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=1000)
    test_size = -3000  # -1 for all
    (x_train, y_train), (x_test, y_test) = (x_train[:test_size], y_train[:test_size]), (
        x_test[:int(test_size / 10)], y_test[:int(test_size / 10)])

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    x = keras.layers.Input((MAX_LEN,))
    y = keras.layers.Embedding(MAX_WORD * 10, 64*8)(x)
    y = PositionalEncoding(maxlen=MAX_LEN)(y)
    y = keras.layers.Dropout(0.1)(y)

    y2 = MultiHeadAttention(8, 64,masking_value=-1e8)([y,y,y])
    y2 = keras.layers.Dropout(0.1)(y2)

    y = AddNorm()([y,y2])

    y2 = PointwiseFFWD()(y)
    y2 = keras.layers.Dropout(0.1)(y2)
    y = AddNorm()([y,y2])

    y = keras.layers.Lambda(lambda y: y, output_shape=lambda s: s)(y) # fix TypeError: Layer flatten does not support masking
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(512)(y)
    y = keras.layers.Dense(46, activation='softmax')(y)

    model = keras.Model(inputs=[x], outputs=[y])
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=tf.train.AdamOptimizer(),
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    model.fit(x_train, y_train, epochs=1)
    model.summary()
    model.evaluate(x_test, y_test)

test()


"""
Epoch 1/1
7982/7982 [==============================] - 318s 40ms/step - loss: 16.0575 - sparse_categorical_accuracy: 0.0038
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 200)          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 200, 512)     51200000    input_1[0][0]                    
__________________________________________________________________________________________________
positional_encoding (Positional (None, 200, 512)     0           embedding[0][0]                  
__________________________________________________________________________________________________
multi_head_attention (MultiHead (None, 200, 512)     0           positional_encoding[0][0]        
                                                                 positional_encoding[0][0]        
                                                                 positional_encoding[0][0]        
__________________________________________________________________________________________________
flatten (Flatten)               (None, 102400)       0           multi_head_attention[0][0]       
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 46)           4710446     flatten[0][0]                    
==================================================================================================
Total params: 55,910,446
Trainable params: 55,910,446
Non-trainable params: 0
__________________________________________________________________________________________________


After adding positionalffwd

Epoch 1/1
5982/5982 [==============================] - 620s 104ms/step - loss: nan - sparse_categorical_accuracy: 0.0057
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 200)          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 200, 512)     51200000    input_1[0][0]                    
__________________________________________________________________________________________________
positional_encoding (Positional (None, 200, 512)     0           embedding[0][0]                  
__________________________________________________________________________________________________
dropout (Dropout)               (None, 200, 512)     0           positional_encoding[0][0]        
__________________________________________________________________________________________________
multi_head_attention (MultiHead (None, 200, 512)     0           dropout[0][0]                    
                                                                 dropout[0][0]                    
                                                                 dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 200, 512)     0           multi_head_attention[0][0]       
__________________________________________________________________________________________________
add_norm (AddNorm)              (None, 200, 512)     204800      dropout[0][0]                    
                                                                 dropout_1[0][0]                  
__________________________________________________________________________________________________
pointwise_ffwd (PointwiseFFWD)  (None, 200, 512)     0           add_norm[0][0]                   
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 200, 512)     0           pointwise_ffwd[0][0]             
__________________________________________________________________________________________________
add_norm_1 (AddNorm)            (None, 200, 512)     204800      add_norm[0][0]                   
                                                                 dropout_2[0][0]                  
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 200, 512)     0           add_norm_1[0][0]                 
__________________________________________________________________________________________________
flatten (Flatten)               (None, 102400)       0           lambda_1[0][0]                   
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 512)          52429312    flatten[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 46)           23598       dense_3[0][0]                    
==================================================================================================
Total params: 104,062,510
Trainable params: 104,062,510
Non-trainable params: 0
__________________________________________________________________________________________________
    
"""

