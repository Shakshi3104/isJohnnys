import json
import os
import urllib

from bs4 import BeautifulSoup
import requests


# パーセントエンコーディング文字列の変換
# 参考: https://qiita.com/yagays/items/e59731b3930252b5f0c4
def parse_japanese(keyword):
    parse_keyword = []
    for word in keyword:
        parse_keyword.append(urllib.parse.quote(word))

    return parse_keyword


# Google画像検索から画像をスクレイピングする
# 参考: https://qiita.com/Jixjia/items/881c03c50c6f07b0b6ab
def get_images_from_google_search(keyword, max_images, output_dir):
    keyword_list = keyword
    if type(keyword) is str:
        keyword_list = [keyword]

    # 保存するディレクトリを作る
    query = "+".join(keyword_list)
    save_directory = output_dir + "/" + query
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # パーセントエンコーディング文字列の変換
    query_ = "+".join(parse_japanese(keyword))

    # スクレイピング
    url = "https://www.google.co.jp/search?q="+query_+"&source=lnms&tbm=isch"
    header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

    soup = BeautifulSoup(urllib.request.urlopen(
        urllib.request.Request(url=url, headers=header)
            ), 'html.parser')

    ActualImages = []
    print("Searching " + query + " images ...")

    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        ActualImages.append((link, Type))
    for i, (img, Type) in enumerate(ActualImages[0:max_images]):
        if Type == 'img':
            continue

        try:
            Type = Type if len(Type) > 0 else 'jpg'
            print("Downloading image {} ({}), type is {}".format(i, img, Type))
            raw_img = urllib.request.urlopen(img).read()
            f = open(os.path.join(save_directory, "img_" + str(i) + "." + Type), 'wb')
            f.write(raw_img)
            f.close()
        except Exception as e:
            print("could not load : " + img)
            print(e)

    return save_directory
