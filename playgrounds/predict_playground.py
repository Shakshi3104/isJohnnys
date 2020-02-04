from tensorflow.keras.models import load_model

from images.predict import predict_images

# OMP Error 対策
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 予測
model = load_model("/Users/user/Downloads/vgg16_imagenet.h5")
predict_dir = "/Users/user/Downloads/predict"
predict_images(predict_dir, model, gradcam=False)
# predict_images(predict_dir, model, gradcam=True)
