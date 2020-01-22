from images.collect import ImageCollector
from images.load import ImageClipper, ImageLoader
from images.dataset import Dataset

# collector = ImageCollector("/Users/user/Downloads/")
# collector.collect_johnnys_images()
# collector.collect_others_images()

# clipper = ImageClipper("/Users/user/Downloads/")
# clipper.clip_johnnys_image()
# clipper.clip_others_image()

# loader = ImageLoader("/Users/user/Downloads/face/", ["相葉雅紀", "松本潤", "二宮和也", "大野智", "櫻井翔"])
# images, labels, details = loader.load_johnnys_images()

dataset = Dataset("/Users/user/Downloads/face/")
images, labels, details = dataset.load_data()
print(images.shape)