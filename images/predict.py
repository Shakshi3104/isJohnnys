import numpy as np
import cv2
import os

from networks.gradcam import grad_cam


def predict_gradcam(image, model, layer_name="block5_conv3"):
    print(image.shape)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = "/Users/user/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

    if len(face_list) > 0:
        for rect in face_list:
            x, y, width, height = rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[3:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            if image.shape[0] < 64:
                print("too small")
                continue
            img_shape = img.shape[0]
            img = cv2.resize(img, (64, 64))

            # grad-cam
            cam = grad_cam(model, img, layer_name)
            cam = cv2.resize(cam, (img_shape, img_shape))
            image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cam

            # モデルの予測結果
            img = img / 255.
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            pred = np.argmax(pred, axis=1)

            if pred == 0:
                label = "Johnny's"
            else:
                label = "Other"

            cv2.putText(image, label, (x, y + height + 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    # 顔が検出されなかったとき
    else:
        print("No face")

    return image


def predict_face(image, model):
    print(image.shape)

    image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

            img = cv2.resize(img, (64, 64))
            img = img / 255.
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


def predict_images(input_dir, model, gradcam=False):
    save_directory = input_dir + "/" + "predicted"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

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
                if gradcam:
                    head = "gradcam_"
                    predict_image = predict_gradcam(image, model)
                else:
                    head = "predict_"
                    predict_image = predict_face(image, model)

                # plt.imshow(predict_image)
                # plt.savefig(save_directory + "/" + head + file)
                # plt.show()

                predict_image = cv2.cvtColor(predict_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_directory + "/" + head + file, predict_image)

            except Exception as e:
                print("could not open : " + filepath + "by `imread`")
                print(e)
