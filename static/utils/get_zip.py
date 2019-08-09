import os
import timeit
import shutil
import re
import tarfile


def select_latest_file(_dir):
    file_list = os.listdir(_dir)
    file_list = [file for file in file_list if file.endswith('.tar')]
    # x.index('_') 시간값 문자열의 시작 넘버
    file_list = sorted(file_list, key=lambda x: x[x.index('_'):], reverse=False)
    return file_list[-1]


def order_list(img_dir):
    file_list = os.listdir(img_dir)
    file_list = [file for file in file_list if file.endswith(('.jpeg', '.png', '.jpg'))]
    file_list = sorted(file_list, key=lambda x: re.sub('[._a-zA-Z]', '', x), reverse=False)
    return file_list


def get_zip(tar_path, tmp_dir):
    start = timeit.default_timer()
    tar_file = select_latest_file(tar_path)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    else:
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
    with tarfile.open(f'{tar_path}/{tar_file}', "r:tar") as tar:
        tar.extractall(tmp_dir)

    img_list = order_list(tmp_dir)
    img_list = [tmp_dir+img for img in img_list]
    stop = timeit.default_timer()
    print("압축풀기", stop - start)
    return img_list