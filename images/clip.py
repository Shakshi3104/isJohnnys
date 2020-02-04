import os
import cv2


def clip_images(input_dir, output_dir):
    input_dir_ = input_dir
    output_dir_ = output_dir

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
                        for i, rect in enumerate(face_list):
                            x, y, width, height = rect
                            # img = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                            img = img[y:y + height, x:x + width]

                            # サイズが64以上ではなかったとき
                            if img.shape[0]<64:
                                continue

                            # resize 64x64
                            img = cv2.resize(img, (64, 64))

                            # save
                            filename = output_dir__ + "/" + str(i) + "-" + file
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


class ImageClipper():
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_johnnys_dir = input_dir + "johnnys/"
        self.input_others_dir = input_dir + "others/"
        self.output_johnnys_dir = input_dir + "face/johnnys/"
        self.output_others_dir = input_dir + "face/others/"

    # 参考: https://qiita.com/nirs_kd56/items/bc78bf2c3164a6da1ded

    def clip_johnnys_image(self):
        clip_images(input_dir=self.input_johnnys_dir, output_dir=self.output_johnnys_dir)

    def clip_others_image(self):
        clip_images(input_dir=self.input_johnnys_dir, output_dir=self.output_others_dir)