from images.search import get_images_from_google_search
from images.collect import ImageCollector
from images.load import ImageLoader
from images.clip import ImageClipper, clip_images
from images.dataset import Dataset
from images.predict import predict_images

# collector = ImageCollector("/Users/user/Downloads/")
# collector.collect_johnnys_images()
# collector.collect_others_images()

# re_collects = [["ジェシー", "SixTONES"],
#                ["ラウール", "Snow", "Man"],
#                ["橋本亮介", "A.B.C-Z"]]
#
# for re_collect in re_collects:
#     get_images_from_google_search(re_collect, 100, "/Users/user/Downloads/Johnnys_re/")

# clipper = ImageClipper("/Users/user/Downloads/")
# clipper.clip_johnnys_image()
# clipper.clip_others_image()

# dataset = Dataset("/Users/user/Downloads/face/")
# images, labels, details = dataset.load_data()
# print(images.shape)


clip_images("/Users/user/Downloads/Johnnys_re/", "/Users/user/Downloads/face_re/")
