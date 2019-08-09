from bs4 import BeautifulSoup
import urllib.request
import requests
import shutil
import os


def get_image():
    dir_path = '/home/ubuntu/flask_server/static/models/research/object_detection/test_images/'
    url = "http://192.168.137.63/html/media/"
    req = requests.get(url)

    # 제대로 받아왔느냐? 확인
    if req.status_code != requests.codes.ok:
        print("서버 응답 오류")
        exit()
    else:
        print("실행 된다.")

    shutil.rmtree(dir_path)
    os.mkdir(dir_path)

    html = BeautifulSoup(req.text, 'html.parser')
    
    links = html.select('td a')
    image_group = set()
    for link in links:
        img_name = link.attrs['href']
        try:
            image_group.add(int(img_name[3:7]))
        except:
            pass
    for link in links:
        img_name = link.attrs['href']
        img_url = url + img_name
        try:
            if int(img_name[3:7]) == list(image_group)[-1]:
                urllib.request.urlretrieve(img_url, dir_path+img_name)
                print(img_url)
        except:
            pass


if __name__ == '__main__':
    get_image()