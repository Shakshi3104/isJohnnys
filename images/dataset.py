import numpy as np
import pandas as pd

from images.load import ImageLoader
from images.collect import get_list_from_txt


class Dataset():
    def __init__(self, input_dir):
        johnnys_detail = get_list_from_txt("lists/johnnys_list")
        others_detail = get_list_from_txt("lists/others_list")
        detail_label_list = johnnys_detail + others_detail
        self.loader = ImageLoader(input_dir=input_dir, detail_label_list=detail_label_list)
        self.group_member = pd.read_csv("lists/group_member.csv")

    def load_data(self):
        johnnys_images, johnnys_labels, johnnys_detail_labels = self.loader.load_johnnys_images()
        others_images, others_labels, others_detail_labels = self.loader.load_others_images()

        images = np.concatenate([johnnys_images, others_images], axis=0)
        labels = np.concatenate([johnnys_labels, others_labels], axis=0)
        detail_labels = np.concatenate([johnnys_detail_labels, others_detail_labels], axis=0)

        return images, labels, detail_labels


