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
IMAGE_QUESE = "image_queue"
BATCH_SIZE = 32
SEVER_SLEEP = 0.25
CLIENT_SLEEP =0.25

# initialize Flask application, Redis, & model
app = flask.Flask(__name__)
db = redis.StrictRedis('localhost',port=6379,db=0)
model = None