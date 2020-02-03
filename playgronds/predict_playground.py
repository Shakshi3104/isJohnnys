from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from images.predict import predict_images
from networks.gradcam import grad_cam

# OMP Error 対策
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 予測
model = load_model("/Users/user/Downloads/vgg16_imagenet.h5")
predict_images("/Users/user/Downloads/predict", model, gradcam=False)

# # Grad-cam
# filepath = "/Users/user/Downloads/grad_cam/img_36 15.18.29.jpg"
# # load an image
# img = Image.open(filepath)
#
# # reshape 64x64
# img = img.resize((64, 64))
#
# # to numpy
# img = np.array(img)
# print(img.shape)
#
# cam = grad_cam(model, img, "block5_conv3")
#
# cv2.imwrite("/Users/user/Downloads/gradcam.jpg", cam)
