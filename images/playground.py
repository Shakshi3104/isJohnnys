from images.collect import ImageCollector
from images.load import ImageClipper, ImageLoader
from images.dataset import Dataset
from images.predict import detect_images
from tensorflow.keras.models import load_model

# collector = ImageCollector("/Users/user/Downloads/")
# collector.collect_johnnys_images()
# collector.collect_others_images()

# clipper = ImageClipper("/Users/user/Downloads/")
# clipper.clip_johnnys_image()
# clipper.clip_others_image()

# dataset = Dataset("/Users/user/Downloads/face/")
# images, labels, details = dataset.load_data()
# print(images.shape)


# from images.dataset import Dataset
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from networks.models import VGG
# from tensorflow.keras.optimizers import Adam
# from networks.training import plot_history
# from images.predict import detect_images
#
# input_dir = "/Users/user/Downloads/face/"
# dataset = Dataset(input_dir=input_dir, colab=False)
# images, labels, detail_labels = dataset.load_data()
#
# x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
#
# y_train = to_categorical(y_train, 2)
# y_test_ = to_categorical(y_test, 2)
#
# model = VGG(weight_layer_num=11, side=64, labels=2)
# model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
# stack = model.fit(x_train, y_train, epochs=10, batch_size=50, validation_data=(x_test, y_test_))
#
# score = model.evaluate(x_test, y_test_, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# plot_history(stack)
#
# detect_images("/Users/user/Downloads/predict", model)
