from PIL import Image
import os
import numpy as np


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
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_johnnys_dir = input_dir + "johnnys/"
        self.input_others_dir = input_dir + "others/"

    # 参考: https://qiita.com/hiroeorz@github/items/ecb39ed4042ebdc0a957
    def load_images(self, johnnys=True):
        label_ = 1
        dirpath = self.input_johnnys_dir
        if not johnnys:
            label_ = 0
            dirpath = self.input_others_dir

        image_list_ = []
        init = True

        for dir_ in dirpath:
            if dir_.startswith("."):
                continue

            dir1 = dirpath + dir_
            tmp_list = []

            for file in os.listdir(dir1):
                if not file.startswith("."):
                    filepath = dir1 + "/" + file
                    print(filepath)

                    # load an image
                    img = Image.open(filepath)

                    """ 顔まわりをクリッピングする処理を書く"""
                    # crop an image
                    img = crop_max_square(img)

                    # reshape 25x25
                    img = img.resize((25, 25))
                    """ ------------------------------ """

                    # to numpy
                    img = np.array(img)

                    tmp_list.append(img / 255.)
                    print(img.shape)

            tmp_list = np.array(tmp_list).reshape(-1, 25, 25, 3)
            if init:
                image_list_ = tmp_list
                init = False
            else:
                image_list_ = np.append(image_list_, tmp_list, axis=0)

        label_list_ = np.array([label_] * len(image_list_))

        # return list of images and list of correct labels
        return image_list_, label_list_

    def load_johnnys_images(self):
        self.load_images(johnnys=True)

    def load_others_images(self):
        self.load_images(johnnys=False)