from PIL import Image
import os
import numpy as np
import unicodedata


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
            # UTF-8-Mac -> UTF-8
            detail_label = unicodedata.normalize('NFC', dir_)

            for file in os.listdir(dir1):
                if not file.startswith("."):
                    # 詳細なラベルのリスト
                    label_detail_list_.append(self.detail_label_list.index(detail_label))

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



