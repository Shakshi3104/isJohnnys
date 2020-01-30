from keras.applications import vgg16
from keras.layers import Flatten, Dense
from keras.models import Sequential, Model

# モデル
vgg_ = vgg16.VGG16(weights=None, input_shape=(64, 64, 3), include_top=False)

top_ = Sequential()
top_.add(Flatten(input_shape=vgg_.output_shape[1:]))
top_.add(Dense(1024, activation='relu'))
top_.add(Dense(1024, activation='relu'))
top_.add(Dense(2, activation='softmax'))

model = Model(inputs=vgg_.inputs, outputs=top_(vgg_.outputs))

# 重みをloadする
model.load_weights("/Users/user/Downloads/vgg16_imagenet.hdf5")
