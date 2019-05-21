# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import backend as K

batch_size = 32
epochs = 200
data_augmentation = True
num_class = 10
substract_pixel_mean = True
n = 3

depth = n*6+2
