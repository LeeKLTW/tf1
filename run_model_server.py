# encoding: utf-8
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from threading import Thread
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io

# initialize constants used to control image spatial dimensions and data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queueing
IMAGE_QUESE = "image_queue"  # key name store in redis
BATCH_SIZE = 32
SEVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# initialize Flask application, Redis, & model
app = flask.Flask(__name__)
db = redis.StrictRedis('localhost', port=6379, db=0)
model = None


def base64_encode_image(a):
    return base64.b64decode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    if sys.version_info.major == 3:
        a = bytes(a, encoding='utf-8')
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    # decodestring() is a deprecated alias since Python 3.1 use decodebytes()
    a = a.reshape(shape)
    return a


def prepare_image(image, target):
    if image.mode != 'RGB':
        image.convert('RGB')

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def classify_process():
    model = ResNet50(weights='imagenet')

    while True:
        queue = db.lrange(IMAGE_QUESE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.load(q.decode('utf-8'))
            image = base64_decode_image(q['image'], IMAGE_DTYPE, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])

            imageIDs.append(q["id"])
