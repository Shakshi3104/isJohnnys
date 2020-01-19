from images.collections import ImageCollector, get_list_from_txt

imgc = ImageCollector("/Users/user/Downloads/")
johnnys_list = get_list_from_txt("lists/johnnys_list")
for men in johnnys_list:
    imgc.collect_images(men, max_images=100, johnnys=True)
