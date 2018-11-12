# 先将照片分为走路图和进门图。将走路图几张和为一组，生成GEI进行训练。判断时也用类似的GEI进行
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_WALK = False
DATA_TRAIN = False
DATA_GEI = True


def remove_ds_store(_path):
    for item in _path:
        try:
            if item == '.DS_Store':
                _path.remove(item)
        except OSError as e:
            print(e)
    return _path


def make_dir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('new folder')
    else:
        print('there exists a folder')


def list_sort(image_list):
    image_list = remove_ds_store(image_list)
    list_num = []
    for i in range(image_list.__len__()):
        list_num.append(int(image_list[i].split('.')[0]))
    list_num.sort()
    if image_list[0].split('.')[-1] == 'jpg':
        list_num = map(lambda x: str(x) + '.jpg', list_num)
    else:
        list_num = map(lambda x: str(x), list_num)
    return list(list_num)


def whether_324(image):
    return np.max([np.sum(img[:, 43] - img[:, 42] > 50),
                   np.sum(img[:, 44] - img[:, 43] > 50),
                   np.sum(img[:, 45] - img[:, 44] > 50)]) > 60


def whether_322(imgage):
    return np.max([np.sum(img[:, 175] - img[:, 174] > 50),
                   np.sum(img[:, 176] - img[:, 175] > 50),
                   np.sum(img[:, 177] - img[:, 176] > 50)]) > 60


if DATA_WALK:
    path = ('./data/' if os.name == 'posix' else 'D:/test_D/data/')
    save_path = ('./data_walk/' if os.name == 'posix' else 'D:/test_D/data_walk/')
    people_list = os.listdir(path)
    people_list.sort()
    for people in people_list:
        round_list = os.listdir(path + people)
        round_list = list_sort(round_list)
        for _round in round_list:
            image_list = os.listdir(path + people + '/' + _round)
            image_list = list_sort(image_list)
            for image in image_list:
                image_path = path + people + '/' + _round + '/' + image
                img = cv2.imread(image_path)
                for channel in cv2.split(img):
                    if np.sum(channel >= 200) > 200:
                        img = channel
                # print(whether_324(img))
                if not whether_322(img) and np.sum(img[:170] > 200) < 50:
                    image_save_path = save_path + people + '/' + _round + '/'
                    make_dir(image_save_path)
                    plt.imsave(image_save_path + image, img, cmap='gray')
                elif np.sum(img[45: 175] > 200) > 200 and not whether_324(img):
                    image_save_path = save_path + people + '/' + _round + '/'
                    make_dir(image_save_path)
                    plt.imsave(image_save_path + image, img, cmap='gray')


if DATA_TRAIN:
    path = ('./data_walk/' if os.name == 'posix' else 'D:/test_D/data_walk/')
    save_path = ('./data_separate/' if os.name == 'posix' else 'D:/test_D/data_separate/')
    people_list = os.listdir(path)
    people_list.sort()
    people_list = remove_ds_store(people_list)
    for people in people_list:
        round_list = os.listdir(path + people)
        round_list = list_sort(round_list)
        round_list = remove_ds_store(round_list)
        round = 1
        for _round in round_list:
            image_list = os.listdir(path + people + '/' + _round)
            image_list = list_sort(image_list)
            image_list = remove_ds_store(image_list)
            if image_list.__len__() <= 5:
                number = 0
                for image in image_list:
                    image_path = path + people + '/' + _round + '/' + image
                    img = cv2.imread(image_path)
                    for channel in cv2.split(img):
                        if np.sum(channel >= 200) > 200:
                            img = channel
                    image_save_path = save_path + people + '/' + str(round) + '/'
                    make_dir(image_save_path)
                    plt.imsave(image_save_path + str(number) + '.jpg', img, cmap='gray')
                    number += 1
                round += 1
            else:
                num = image_list.__len__() // 5
                rest = image_list.__len__() % 5
                if rest > 2:
                    num = num + 1
                for i in range(num):
                    if i != num - 1:
                        number = 1
                        for image in image_list[5*i: 5*i+5]:
                            image_path = path + people + '/' + _round + '/' + image
                            img = cv2.imread(image_path)
                            for channel in cv2.split(img):
                                if np.sum(channel >= 200) > 200:
                                    img = channel
                            image_save_path = save_path + people + '/' + str(round) + '/'
                            make_dir(image_save_path)
                            plt.imsave(image_save_path + str(number) + '.jpg', img, cmap='gray')
                            number += 1
                        round += 1
                    else:
                        number = 1
                        for image in image_list[5*i:]:
                            image_path = path + people + '/' + _round + '/' + image
                            img = cv2.imread(image_path)
                            for channel in cv2.split(img):
                                if np.sum(channel >= 200) > 200:
                                    img = channel
                            image_save_path = save_path + people + '/' + str(round) + '/'
                            make_dir(image_save_path)
                            plt.imsave(image_save_path + str(number) + '.jpg', img, cmap='gray')
                            number += 1
                        round += 1


if DATA_GEI:
    path = ('./data_separate/' if os.name == 'posix' else 'D:/test_D/data_separate/')
    save_path = ('./data_separate_GEI/' if os.name == 'posix' else 'D:/test_D/data_separate_GEI/')
    people_list = os.listdir(path)
    people_list.sort()
    people_list = remove_ds_store(people_list)
    for people in people_list:
        round_list = os.listdir(path + people)
        round_list = list_sort(round_list)
        round_list = remove_ds_store(round_list)
        round = 1
        for _round in round_list:
            image_list = os.listdir(path + people + '/' + _round)
            image_list = list_sort(image_list)
            image_list = remove_ds_store(image_list)











