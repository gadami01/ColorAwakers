import os
import shutil
import xlsxwriter
import numpy as np
import random
import os
import cv2

# Define the base path for the source dataset
train_path = "C:/Users/George/Desktop/imagenet-mini/train"
val_path = "C:/Users/George/Desktop/imagenet-mini/val"

# Define the base path for the destination folders
dest_path = "C:/Users/George/Desktop/imagenet-mini-split-128"
train_dest_path = "C:/Users/George/Desktop/imagenet-mini-split-128/train"
val_dest_path = "C:/Users/George/Desktop/imagenet-mini-split-128/val"
test_dest_path = "C:/Users/George/Desktop/imagenet-mini-split-128/test"

os.makedirs(dest_path, exist_ok=True)
os.makedirs(train_dest_path, exist_ok=True)
os.makedirs(val_dest_path, exist_ok=True)
os.makedirs(test_dest_path, exist_ok=True)

trainingCounter = 0
validationCounter = 0
testCounter = 0

for imgDir in os.listdir(train_path):
    for img in os.listdir(train_path + "/" + imgDir):
        img = cv2.imread(train_path + "/" + imgDir + "/" + img)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(train_dest_path, str(trainingCounter) + '.JPEG'), img)
        trainingCounter += 1

for imgDir in os.listdir(val_path):
    for img in os.listdir(val_path + "/" + imgDir):
        if random.random() >= 0.5:
            img = cv2.imread(val_path + "/" + imgDir + "/" + img)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(val_dest_path, str(validationCounter) + '.JPEG'), img)
            validationCounter += 1
        else:
            img = cv2.imread(val_path + "/" + imgDir + "/" + img)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(test_dest_path, str(testCounter) + '.JPEG'), img)
            testCounter += 1




