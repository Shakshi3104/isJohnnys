from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import vgg16, vgg19
# from tensorflow.image import resize_images         # tensorflow v1.14
from tensorflow.keras.backend import resize_images   # tensorflow v1.15


def VGG(weight_layer_num=16, side=64, labels=2):
    "weight_layer_num = 11 or 13 or 16 or 19, それ以外の数字の場合は11になる"

    inputs = Input(shape=(side, side, 3))
    # 画像のサイズを224x224にする
    # http://blog.neko-ni-naritai.com/entry/2018/04/07/115504を参考にした
    x = Lambda(lambda image: resize_images(image, (224, 224)))(inputs)

    # conv block 1
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    if weight_layer_num == 13 or weight_layer_num == 16 or weight_layer_num == 19:
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # conv block 2
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    if weight_layer_num == 13 or weight_layer_num == 16 or weight_layer_num == 19:
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # conv block 3
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    if weight_layer_num == 16 or weight_layer_num == 19:
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    if weight_layer_num == 19:
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # conv block 4 and conv block 5
    for _ in range(0, 2):
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
        if weight_layer_num == 16 or weight_layer_num == 19:
            x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
        if weight_layer_num == 19:
            x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dense(units=4096, activation='relu')(x)
    predictions = Dense(labels, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    return model


def pretrained_VGG(weight_layer_num=16, side=64, labels=2, frozen_layer_num=15):
    if weight_layer_num == 19:
        vgg_ = vgg19.VGG19(weights='imagenet', input_shape=(side, side, 3), include_top=False)
    else:
        vgg_ = vgg16.VGG16(weights='imagenet', input_shape=(side, side, 3), include_top=False)

    top_ = Sequential()
    top_.add(Flatten(input_shape=vgg_.output_shape[1:]))
    top_.add(Dense(4096, activation='relu'))
    top_.add(Dense(4096, activation='relu'))
    top_.add(Dense(labels, activation='softmax'))

    model = Model(inputs=vgg_.input, output=top_(vgg_.output))

    for layer in vgg_.layers[:frozen_layer_num]:
        layer.trainable = False

    return model
