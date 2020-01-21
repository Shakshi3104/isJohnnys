from images.collect import ImageCollector
from images.load import ImageClipper, ImageLoader

collector = ImageCollector("/Users/user/Downloads/")
collector.collect_johnnys_images(max_images=15)

clipper = ImageClipper("/Users/user/Downloads/")
clipper.clip_johnnys_image()

loader = ImageLoader("/Users/user/Downloads/face/", ["相葉雅紀", "松本潤", "二宮和也", "大野智", "櫻井翔"])
images, labels, details = loader.load_johnnys_images()
