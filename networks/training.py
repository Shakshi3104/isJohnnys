import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

from networks.models import VGG, pretrained_VGG


def plot_history(stack, filename=None):
    sns.set_context("poster")
    plt.figure(figsize=(10, 12))

    epochs = len(stack.history['accuracy'])
    e = range(epochs)

    plt.subplot(2, 1, 1)
    sns.lineplot(x=e, y=stack.history['accuracy'], label="acc", color='darkcyan')
    sns.lineplot(x=e, y=stack.history['val_accuracy'], label='val_acc', color='coral')
    plt.title("accuracy")
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    sns.lineplot(x=e, y=stack.history['loss'], label="loss", color='darkcyan')
    sns.lineplot(x=e, y=stack.history['val_loss'], label='val_loss', color='coral')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.legend(loc='best')

    if filename is not None:
        plt.savefig(filename)

    plt.show()
    plt.figure()


def training(images, labels, pretrain=False, history_path=None, epochs=100, batch_size=50, lr=1e-4, frozen_layer_num=15):
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

    y_train = to_categorical(y_train, 2)
    y_test_ = to_categorical(y_test, 2)

    if pretrain:
        model = pretrained_VGG(weight_layer_num=16, side=64, labels=2, frozen_layer_num=frozen_layer_num)
    else:
        model = VGG(weight_layer_num=16, side=64, labels=2)

    model.compile(optimizer=Adam(lr=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    stack = model.fit(x_train, y_train,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(x_test, y_test_))

    score = model.evaluate(x_test, y_test_, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plot_history(stack, filename=history_path)

    return model


class ConfusionMatrix(Callback):
    def __init__(self, model, x, y, label_list=None):
        """
        :parameter
        model: tensorflow.kerasのモデル
        x: np.ndarray
        y: np.ndarray, not one-hot vector
        label_list: list(string), optional, list of label name
        """
        self.model = model
        self.x_val = x
        self.y_val = y
        self.label_list = label_list

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            predict = self.model.predict(self.x_val)
            predict = np.argmax(predict, axis=1)
            cf = confusion_matrix(predict, self.y_val)

            if self.label_list is not None:
                cf = pd.DataFrame(cf, columns=self.label_list, index=self.label_list)

            print("")
            print("Confusion Matrix")
            print(cf)
