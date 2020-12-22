from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import vgg16, vgg19, mobilenet, mobilenet_v2, resnet#, efficientnet


def VGG(weight_layer_num=16, side=64, labels=2):
    "weight_layer_num = 11 or 13 or 16 or 19, それ以外の数字の場合は11になる"

    inputs = Input(shape=(side, side, 3))
    # 画像のサイズを224x224にする
    # http://blog.neko-ni-naritai.com/entry/2018/04/07/115504を参考にした
    # x = Lambda(lambda image: resize_images(image, height_factor=3, width_factor=3, data_format="channels_last"))(inputs)

    # conv block 1
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_initializer='he_normal')(inputs)
    if weight_layer_num == 13 or weight_layer_num == 16 or weight_layer_num == 19:
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # conv block 2
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_initializer='he_normal')(x)
    if weight_layer_num == 13 or weight_layer_num == 16 or weight_layer_num == 19:
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # conv block 3
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_initializer='he_normal')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_initializer='he_normal')(x)
    if weight_layer_num == 16 or weight_layer_num == 19:
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(x)
    if weight_layer_num == 19:
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # conv block 4 and conv block 5
    for _ in range(0, 2):
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                   kernel_initializer='he_normal')(x)
        if weight_layer_num == 16 or weight_layer_num == 19:
            x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer='he_normal')(x)
        if weight_layer_num == 19:
            x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer='he_normal')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(labels, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    return model


def pretrained_VGG(weight_layer_num=16, side=64, labels=2, frozen_layer_num=None, frozen_block_num=None):
    if weight_layer_num == 19:
        vgg_ = vgg19.VGG19(weights='imagenet', input_shape=(side, side, 3), include_top=False)
        block_layer_table = [0, 4, 7, 12, 17, 21]
    else:
        vgg_ = vgg16.VGG16(weights='imagenet', input_shape=(side, side, 3), include_top=False)
        block_layer_table = [0, 4, 7, 11, 15, 19]

    top_ = Sequential()
    top_.add(Flatten(input_shape=vgg_.output_shape[1:]))
    top_.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
    top_.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
    top_.add(Dense(labels, activation='softmax'))

    model = Model(inputs=vgg_.inputs, outputs=top_(vgg_.outputs))

    if frozen_layer_num is None and frozen_block_num is None:
        print("No frozen layers: fine-tuning")
        return model

    if frozen_block_num is not None:
        frozen_layer_num = block_layer_table[frozen_block_num]

    for layer in vgg_.layers[:frozen_layer_num]:
        layer.trainable = False

    for i, layer in enumerate(model.layers):
        print("layer " + str(i) + " trainable: " + str(layer.trainable))

    return model


class PretrainedModel:
    def __init__(self, model_name="VGG16", input_shape=(64, 64, 3), extractor=False):
        self.model_name = model_name
        self.input_shape = input_shape
        self.extractor = extractor

    def __call__(self, *args, **kwargs):
        if self.model_name in ["VGG16", "vgg16"]:
            pre_trained = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)

        elif self.model_name in ["VGG19", "vgg19"]:
            pre_trained = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)

        elif self.model_name in ["MobileNet", "mobilenet"]:
            pre_trained = mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=self.input_shape)

        elif self.model_name in ["MobileNetV2", "MobileNet_V2", "mobilenetv2", "mobilenet_v2", "mobilenet_V2"]:
            pre_trained = mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=self.input_shape)

        elif self.model_name in ["resnet50", "ResNet50"]:
            pre_trained = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)

        # elif self.model_name in ["EfficientNetB0", "efficientnetb0"]:
        #     pre_trained = efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.input_shape)
        #
        # elif self.model_name in ["EfficientNetB5", "efficientnetb5"]:
        #     pre_trained = efficientnet.EfficientNetB5(include_top=False, weights='imagenet', input_shape=self.input_shape)

        else:
            print("Not exists {}".format(self.model_name))
            return None

        if self.extractor:
            for layer in pre_trained.layers:
                layer.trainable = False

        if self.model_name in ["VGG16", "vgg16", "VGG19", "vgg19"]:
            x = Flatten()(pre_trained.output)
            x = Dense(1024, activation="relu", kernel_initializer="he_normal")(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation="relu", kernel_initializer="he_normal")(x)
            x = Dropout(0.5)(x)

        else:
            x = GlobalAveragePooling2D()(pre_trained.output)
            x = Flatten()(x)

        y = Dense(2, activation="softmax")(x)

        model = Model(inputs=pre_trained.input, outputs=y)
        return model


if __name__ == "__main__":
    model = PretrainedModel()()
    print(model.summary())
