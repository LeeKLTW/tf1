# -*- coding: utf-8 -*-
import numpy as np

# cifar
def normalize(x):
    x = x.astype('float32')/255
    return x

def substract_pixel_mean(x):
    mean = np.mean(x)
    x -= mean
    return x
#