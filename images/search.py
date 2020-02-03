import json
import os
import urllib

from bs4 import BeautifulSoup
import requests
import tweepy

from images.utils import get_list_from_txt


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
    # header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0"}

    soup = BeautifulSoup(urllib.request.urlopen(
        urllib.request.Request(url=url, headers=header)
            ), 'html.parser')

    ActualImages = []
    print("Searching " + query + " images ...")

    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        ActualImages.append((link, Type))

    for i, (img, Type) in enumerate(ActualImages[0:max_images]):
        if Type == 'img' or Type == 'l':
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


# Twitterの検索から画像を持ってくる
# 参考: https://qiita.com/koki-sato/items/c16c8e3445287698b3a8
def get_image_urls_from_twitter_search(keyword, max_images=100, pages=10):
    # 認証キー
    key = get_list_from_txt("lists/key")
    CONSUMER_KEY = key[0]
    CONSUMER_SECRET = key[1]
    ACESS_TOKEN = key[2]
    ACESS_TOKEN_SECRET = key[3]

    # OAuth認証
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACESS_TOKEN, ACESS_TOKEN_SECRET)

    # APIインスタンスを生成
    api = tweepy.API(auth)

    # 検索
    max_id = None
    media_url_list = []
    search_word = "\"" + keyword + "\" lang:ja filter:images"
    for _ in range(pages):
        if max_id is None:
            search_results = api.search(q=search_word, rpp=max_images)
        else:
            search_results = api.search(q=search_word, rpp=max_images, max_id=max_id)
        # URLを取得
        for result in search_results:
            if 'media' in result.entities:
                for media in result.entities['media']:
                    url = media['media_url_https']
                    if url not in media_url_list:
                        media_url_list.append(url)
        max_id = result.id-1

    return media_url_list


def get_images_from_twitter_search(keyword, max_images, output_dir):
    media_url_list = get_image_urls_from_twitter_search(keyword, max_images)

    save_directory = output_dir + keyword
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    print("Searching " + keyword + " images ...")

    for url in media_url_list:
        # print(url)
        url_orig = url + ":orig"
        filename = url.split('/')[-1]
        savepath = save_directory + "/" + filename

        try:
            print("Downloading images from " + url_orig)
            raw_img = urllib.request.urlopen(url_orig).read()
            f = open(os.path.join(savepath), 'wb')
            f.write(raw_img)
            f.close()
        except Exception as e:
            print("could not load : " + url)
            print(e)


if __name__ == '__main__':
    research_list = ["藤ヶ谷大輔", "ジェシー"]

    for word in research_list:
        get_images_from_twitter_search(word, 100, "/Users/user/Downloads/Johnnys_re/")
