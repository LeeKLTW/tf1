# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding,GlobalAveragePooling1D,Dropout, Dense
from attention import Attention

q_list = []
a_list = []

with open('data/FAQ.csv','r') as f:
    for line in f:
        l = line.split(',')
        if len(l)==3:
            q,a,_ = l
        elif len(l)==2:
            q,a = l
        q_list.append(q)
        a_list.append(a)


batch_size = 8
NUM_WORDS = 3000
MAX_LEN = 200

print('max length quesiton length',max([len(i) for i in q_list]))
print('max length quesiton length',max([len(a) for i in a_list]))
print('max length used in model',MAX_LEN)
print('number of words',NUM_WORDS)

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = q_list
y_train = a_list

#todo build test set
x_test = None
y_test = None

tokenizer = Tokenizer(num_words=NUM_WORDS, char_level=True, oov_token=1)
tokenizer.fit_on_texts(''.join(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)


y_train = a_list
y_train = tokenizer.texts_to_sequences(y_train)
y_train = keras.preprocessing.sequence.pad_sequences(y_train, maxlen=MAX_LEN)



q_inputs = Input(shape=(None,), dtype='int32')
a_inputs = Input(shape=(None,), dtype='int32')

embeddings = Embedding(NUM_WORDS, 128)(q_inputs)
embeddings = Embedding(NUM_WORDS, 128)(a_inputs)

O_seq = Attention(16,16)([embeddings,embeddings,embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)

outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=[q_inputs,a_inputs], outputs=outputs)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

print('Train...')
hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))

model.evaluate(x_test, y_test)
