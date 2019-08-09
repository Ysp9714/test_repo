from bs4 import BeautifulSoup
import urllib.request
import requests
import shutil
import os
import paramiko
import webbrowser

def ftp_send(img_names):
    host = '222.122.196.217'
    port = 16022
    transport = paramiko.Transport((host, port))

    username = "ubuntu"
    key = "NGsKvKJWJNazleL"
    transport.start_client()
    transport.auth_password(username=username, password=key)
    sftp = paramiko.SFTPClient.from_transport(transport)

    for img_name in img_names:
        filepath = '/home/ubuntu/flask_server/static/models/research/object_detection/test_images/'
        localpath = 'C:/Users/ysp97/Desktop/test/'
        sftp.put(localpath+img_name, filepath+img_name[:32])

    sftp.close()
    transport.close()


def get_image():
    url = "http://192.168.137.63/html/media/"
    req = requests.get(url)
    dir_path = "C:/Users/ysp97/Desktop/test/"
    img_names = list()

    # 제대로 받아왔느냐? 확인
    if req.status_code != requests.codes.ok:
        print("서버 응답 오류")
        exit()

    html = BeautifulSoup(req.text, 'html.parser')
    shutil.rmtree(dir_path)
    os.mkdir(dir_path, mode=0o777)

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
                urllib.request.urlretrieve(img_url, dir_path + img_name)
                img_names.append(img_name)
                print(img_name)
        except:
            pass
    return img_names


if __name__ == '__main__':
    url = 'http://222.122.196.217:5005/ready_to_prediction'
    req = requests.get(url)
    ftp_send(get_image())
    url = 'http://222.122.196.217:5005/detection_test'
    webbrowser.open(url)