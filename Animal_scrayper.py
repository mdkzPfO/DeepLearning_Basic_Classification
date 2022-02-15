import os
import time
from selenium import webdriver
from PIL import Image
import io
import requests
import hashlib
import chromedriver_binary
import mimetypes

# クリックなど動作後に待つ時間(秒)
sleep_between_interactions = 2
# ダウンロードする枚数
download_num = 800
# 検索ワード
query = "cat"
# 画像検索用のurl
search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

# サムネイル画像のURL取得
wd = webdriver.Chrome()
wd.get(search_url.format(q=query))
# サムネイル画像のリンクを取得(ここでコケる場合はセレクタを実際に確認して変更する)
thumbnail_results = wd.find_elements_by_css_selector("img.rg_i")

# サムネイルをクリックして、各画像URLを取得
image_urls = set()
for img in thumbnail_results[:download_num]:
    try:
        img.click()
        time.sleep(sleep_between_interactions)
    except Exception:
        continue
    # 一発でurlを取得できないので、候補を出してから絞り込む(やり方あれば教えて下さい)
    # 'n3VNCb'は変更されることあるので、クリックした画像のエレメントをみて適宜変更する
    url_candidates = wd.find_elements_by_class_name('n3VNCb')
    for candidate in url_candidates:
        url = candidate.get_attribute('src')
        if url and 'https' in url:
            image_urls.add(url)
# 少し待たないと正常終了しなかったので3秒追加
time.sleep(sleep_between_interactions+3)
wd.quit()

# 画像のダウンロード
image_save_folder_path = 'data/cat'
print(image_urls)
for url in image_urls:
    print("roop")
    print(url)
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(image_save_folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        print(file_path)
        mime = mimetypes.guess_type(file_path)
        print(mime)
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=90)
        print(f"SUCCESS - saved {url} - as {file_path}")
        print("test03")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")
