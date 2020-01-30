from keras.applications import vgg16
from keras.layers import Flatten, Dense
from keras.models import Sequential, Model

from tensorflow.keras.models import load_model

from images.predict import predict_images

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = load_model("/Users/user/Downloads/vgg16.h5")

predict_images("/Users/user/Downloads/predict", model)
