import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def predict_face(image, model):
    print(image.shape)

    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = "/Users/user/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

    if len(face_list) > 0:
        for rect in face_list:
            x, y, width, height = rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[3:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            if image.shape[0] < 64:
                print("too small")
                continue

            img = cv2.resize(image, (64, 64))
            img = np.expand_dims(img, axis=0)

            # モデルの予測結果
            pred = model.predict(img)
            pred = np.argmax(pred, axis=1)
            if pred == 0:
                label = "Johnny's"
            else:
                label = "Other"

            cv2.putText(image, label, (x, y+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    # 顔が検出されなかったとき
    else:
        print("No face")

    return image


def predict_images(input_dir, model):
    for file in os.listdir(input_dir):
        if not file.startswith("."):
            filepath = input_dir + "/" + file
            print(filepath)

            try:
                image = cv2.imread(filepath)
                if image is None:
                    print("Not open " + filepath)
                    continue

                b, g, r = cv2.split(image)
                image = cv2.merge([r, g, b])
                predict_image = predict_face(image, model)

                plt.imshow(predict_image)
                plt.savefig(input_dir + "/" + "predict_" + file)
                plt.show()

            except Exception as e:
                print("could not open : " + filepath + "by `imread`")
                print(e)
