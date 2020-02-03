import os
from images.search import get_images_from_google_search
from images.utils import get_list_from_txt


# ディレクトリが空かどうかを判定する
def is_empty(directory):
    files = os.listdir(directory)
    files = [f for f in files if not f.startswith(".")]
    if not files:
        return True
    else:
        return False
    # return not Files


class ImageCollector():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_johnnys_dir = output_dir + "johnnys/"
        self.output_others_dir = output_dir + "others/"

    def collect_images(self, men, max_images=100, johnnys=True):
        directory = self.output_johnnys_dir
        if not johnnys:
            directory = self.output_others_dir

        save_dir = get_images_from_google_search(keyword=men, max_images=max_images,
                                                 output_dir=directory)
        # 空だったらもう一回
        if is_empty(save_dir):
            get_images_from_google_search(keyword=men, max_images=max_images,
                                          output_dir=directory)

    def collect_johnnys_images(self, max_images=100):
        johnnys_list = get_list_from_txt("lists/johnnys_list")
        for men in johnnys_list:
            self.collect_images(men=men, max_images=max_images, johnnys=True)

    def collect_others_images(self, max_images=100):
        others_list = get_list_from_txt("lists/others_list")
        for men in others_list:
            self.collect_images(men=men, max_images=max_images, johnnys=False)
