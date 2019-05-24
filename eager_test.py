# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()
tf.executing_eagerly()

BATCH_SIZE = 8


def fizzbuzz(max_num):
    max_num = tf.convert_to_tensor(max_num)  # ? when should I use this, if I already have tf.constant?
    for num in range(max_num.numpy()):
        counter = tf.constant(num)
        if (int(counter.numpy()) % 3 == 0) and (int(counter.numpy()) % 5 == 0):
            print('Fizzbuzz@', int(counter.numpy()))

        elif (int(counter.numpy()) % 3 == 0):
            print('Fizz@', int(counter.numpy()))

        elif (int(counter.numpy()) % 5 == 0):
            print('Buzz@', int(counter.numpy()))
        else:
            print(int(counter.numpy()))


# Eager training
w = tf.Variable([[2]], dtype='float32')

with tf.GradientTape() as tape:
    loss = w * w
    gradient = tape.gradient(loss, w)  # compute loss by applying gradient to w, should be 2*w.
# gradient = tape.gradient(loss, w)  # compute loss by applying gradient to w, should be 2*w.




(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = tf.cast(tf.reshape(x_train, (-1, 28, 28, 1)) / 255., tf.float32)
y_train = tf.cast(tf.reshape(y_train, (-1, 1)), tf.int64) # tf.int64

x_test = tf.cast(tf.reshape(x_train, (-1, 28, 28, 1)) / 255., tf.float32)
y_test = tf.cast(tf.reshape(y_train, (-1, 1)), tf.int64) # tf.int64

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=10).batch(BATCH_SIZE)

inputs = keras.layers.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.Flatten()(x)
y = keras.layers.Dense(10, activation='softmax')(x)

optimizer = tf.train.AdamOptimizer()
model = keras.models.Model(inputs=[inputs], outputs=[y])
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test))

for images, labels in dataset.take(1):  # take 1 = BATCH_SIZE
    logits = model(images).numpy()
    print(logits)

loss_history = []
optimizer = tf.train.AdamOptimizer()
for (batch_cnt, (x_train, y_train)) in enumerate(dataset.take(100)):
    if (batch_cnt + 1) % 10 == 0:
        print('Batch no.', batch_cnt + 1)

    with tf.GradientTape() as tape:
        logits = model(x_train,training=True) # Make sure trainig=True, No need to numpy()
        loss = tf.losses.sparse_softmax_cross_entropy(y_train, logits)
        loss_history.append(loss.numpy())
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())

print(loss_history)

# todo checkout custom gradient in the future