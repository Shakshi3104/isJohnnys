import os
from images.search import get_images_from_google_search


# txtファイルからリストを読み込む
def get_list_from_txt(filepath: str):
    f = open(filepath)
    lines = f.readlines()
    f.close()

    list_ = []
    for line in lines:
        list_.append(line.replace("\n", ""))

    return list_


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
        self.output_not_johnnys_dir = output_dir + "not/"

    def collect_images(self, men, max_images=100, johnnys=True):
        directory = self.output_johnnys_dir
        if not johnnys:
            directory = self.output_not_johnnys_dir

        save_dir = get_images_from_google_search(keyword=men, max_images=max_images,
                                                 output_dir=directory)
        # 空だったらもう一回
        if is_empty(save_dir):
            get_images_from_google_search(keyword=men, max_images=max_images,
                                          output_dir=directory)
