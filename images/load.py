from PIL import Image
import os
import numpy as np
import cv2


# Cropping Center of Image
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


# Cropping Max Squrare Image
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


class ImageLoader():
    def __init__(self, input_dir, detail_label_list):
        self.input_dir = input_dir
        self.input_johnnys_dir = input_dir + "johnnys/"
        self.input_others_dir = input_dir + "others/"
        self.detail_label_list = detail_label_list

    # 参考: https://qiita.com/hiroeorz@github/items/ecb39ed4042ebdc0a957
    def load_images(self, johnnys=True):
        label_ = 1
        dirpath = self.input_johnnys_dir
        if not johnnys:
            label_ = 0
            dirpath = self.input_others_dir

        image_list_ = []
        label_detail_list_ = []
        init = True

        for dir_ in os.listdir(dirpath):
            if dir_.startswith("."):
                continue

            dir1 = dirpath + dir_
            tmp_list = []

            for file in os.listdir(dir1):
                if not file.startswith("."):
                    # 詳細なラベルのリスト
                    label_detail_list_.append(self.detail_label_list.index(dir_))

                    filepath = dir1 + "/" + file
                    print(filepath)

                    # load an image
                    img = Image.open(filepath)

                    # reshape 64x64
                    img = img.resize((64, 64))

                    # to numpy
                    img = np.array(img)

                    tmp_list.append(img / 255.)
                    print(img.shape)

            tmp_list = np.array(tmp_list).reshape(-1, 64, 64, 3)
            if init:
                image_list_ = tmp_list
                init = False
            else:
                image_list_ = np.append(image_list_, tmp_list, axis=0)

        label_list_ = np.array([label_] * len(image_list_))
        label_detail_list_ = np.array(label_detail_list_)

        # return list of images and list of correct labels
        return image_list_, label_list_, label_detail_list_

    def load_johnnys_images(self):
        return self.load_images(johnnys=True)

    def load_others_images(self):
        return self.load_images(johnnys=False)


# 顔まわりをクリッピングして保存する
class ImageClipper():
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_johnnys_dir = input_dir + "johnnys/"
        self.input_others_dir = input_dir + "others/"
        self.output_johnnys_dir = input_dir + "face/johnnys/"
        self.output_others_dir = input_dir + "face/others/"

    # 参考: https://qiita.com/nirs_kd56/items/bc78bf2c3164a6da1ded
    def clip_images(self, johnnys=True):
        input_dir_ = self.input_johnnys_dir
        output_dir_ = self.output_johnnys_dir

        if not johnnys:
            input_dir_ = self.input_others_dir
            output_dir_ = self.output_others_dir

        # 出力用ディレクトリ
        if not os.path.exists(output_dir_):
            os.makedirs(output_dir_)

        for dir_ in os.listdir(input_dir_):
            if dir_.startswith("."):
                continue

            # 入力ディレクトリ
            dir1 = input_dir_ + dir_
            # 出力ディレクトリ
            output_dir__ = output_dir_ + dir_
            if not os.path.exists(output_dir__):
                os.makedirs(output_dir__)

            for file in os.listdir(dir1):
                if not file.startswith("."):
                    filepath = dir1 + "/" + file
                    print(filepath)

                    # load an image
                    try:
                        img = cv2.imread(filepath)

                        if img is None:
                            print("Not open " + filepath)
                            continue

                        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # write Your cascade path
                        cascade_path = "/Users/user/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
                        cascade = cv2.CascadeClassifier(cascade_path)

                        # face recognition
                        face_list = cascade.detectMultiScale(img_gs, scaleFactor=1.1,
                                                             minNeighbors=2, minSize=(64, 64))

                        # 顔が1つ以上検出されたとき
                        if len(face_list) > 0:
                            for rect in face_list:
                                x, y, width, height = rect
                                # img = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                                img = img[y:y + height, x:x + width]

                                # サイズが64以上ではなかったとき
                                if img.shape[0]<64:
                                    continue

                                # resize 64x64
                                img = cv2.resize(img, (64, 64))

                                # save
                                filename = output_dir__ + "/" + file
                                cv2.imwrite(filename, img)
                                print("Saving " + filename)
                                print(img.shape)

                        # 顔が検出されなかったとき
                        else:
                            print("No face")
                            continue

                    except Exception as e:
                        print("could not open : " + filepath + "by `imread`")
                        print(e)

    def clip_johnnys_image(self):
        self.clip_images(johnnys=True)

    def clip_others_image(self):
        self.clip_images(johnnys=False)
