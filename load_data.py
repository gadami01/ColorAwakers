from __future__ import print_function

import os
import random

import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 128

train = []
val = []
test = []

data_path = "C:/Users/George/Desktop/imagenet-mini-split-128/"

def GrayscaleTo3Channels(img):
    img = np.array(Image.fromarray(img, 'L').convert('RGB'))

    return img

def RGBtoGrayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    #
    # plt.imshow(gray, cmap="gray")
    # plt.show()
    # exit(0)

    return gray

def load_train_data():
    train_data_path = data_path + 'train'
    images = os.listdir(train_data_path)
    total = len(images)

    sample = random.sample(list(range(0, total-1)), 20000)

    imgs = np.ndarray((len(sample), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    imgs_gray = np.ndarray((len(sample), IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    i = 0
    j = 0
    print('-' * 30)
    print('Loading training images...')
    print('-' * 30)

    for image_name in images:
        if i in sample:
            img = cv2.imread(train_data_path + "/" + image_name, cv2.IMREAD_COLOR)
            img_gray = RGBtoGrayscale(img)

            imgs[j] = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            # imgs_gray[j] = GrayscaleTo3Channels(img_gray)

            if j % 100 == 0:
                print('Done: {0}/{1} images'.format(j, len(sample)))

            j += 1
        i += 1

    print(imgs_gray.shape)

    return np.array(imgs_gray), np.array(imgs)

def load_val_data():
    validation_data_path = data_path + 'val'
    images = os.listdir(validation_data_path)
    total = len(images)

    imgs = np.ndarray((total, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    imgs_gray = np.ndarray((total, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Loading validation images...')
    print('-' * 30)

    for image_name in images:
        img = cv2.imread(validation_data_path + "/" + image_name, cv2.IMREAD_COLOR)
        img_gray = RGBtoGrayscale(img)

        imgs[i] = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        # imgs_gray[i] = GrayscaleTo3Channels(img_gray)

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    return np.array(imgs_gray), np.array(imgs)

def load_test_data():
    test_data_path = data_path + 'test'
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    imgs_id = []

    i = 0
    print('-' * 30)
    print('Loading test images...')
    print('-' * 30)

    for image_name in images:
        img = cv2.imread(test_data_path + "/" + image_name, cv2.IMREAD_COLOR)

        img_gray = RGBtoGrayscale(img)
        imgs[i] = img_gray
        # imgs[i] = GrayscaleTo3Channels(img_gray)
        imgs_id.append(image_name[:-4] + "_pred")

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    return imgs, imgs_id
