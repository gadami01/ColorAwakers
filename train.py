from __future__ import print_function

import os
from time import time

import cv2
import tensorflow
import numpy as np
from tensorflow.keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf;

import tensorflow as tf;

from SSIM_metric import ciede2000

tf.image
from load_data import load_test_data, load_train_data, load_val_data
import tensorflow.keras.layers as L
tf.image

# from augmentation import augmentation
from keras_unet_collection import models

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.compat.v1.Session(config=config)

K.set_image_data_format('channels_last')  # tensorflow dimension ordering in this code

img_rows = 128
img_cols = 128

e = K.epsilon()
ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5
smooth = 1e-5

def rgb_pixelwise_mse(y_true, y_pred):
    # Calculate the mean squared error (MSE) for each channel separately
    mse_y = tf.math.scalar_mul(1.0,tf.reduce_mean(tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0])))
    mse_cb = tf.reduce_mean(tf.square(y_true[:, :, :, 1] - y_pred[:, :, :, 1]))
    mse_cr = tf.reduce_mean(tf.square(y_true[:, :, :, 2] - y_pred[:, :, :, 2]))

    # Compute the mean MSE across all channels
    mean_mse = (mse_y + mse_cb + mse_cr) / 3.0

    return -mean_mse
    # mse_per_pixel = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    #
    # # Compute the mean MSE across all pixels
    # mean_mse = tf.reduce_mean(mse_per_pixel)
    #
    # return -mean_mse

def rgb_pixelwise_loss(y_true, y_pred):
    return -rgb_pixelwise_mse(y_true, y_pred)

def UNet():
    Xinpt = L.Input([None, None, 1])
    X0 = L.Conv2D(64, (3, 3), padding='same')(Xinpt)
    X0 = L.BatchNormalization()(X0)
    X0 = L.LeakyReLU(alpha=0.2)(X0)  # l,b,64
    X0 = L.Conv2D(64, (3, 3), strides=1, padding='same')(X0)
    X0 = L.BatchNormalization()(X0)
    X0 = L.LeakyReLU(alpha=0.2)(X0)  # l,b,64

    X1 = L.MaxPool2D((2, 2), strides=2)(X0)  # l/2,b/2,64
    X1 = L.Conv2D(128, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.LeakyReLU(alpha=0.2)(X1)
    X1 = L.Conv2D(128, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.LeakyReLU(alpha=0.2)(X1)  # l/2,b/2,128

    X2 = L.MaxPool2D((2, 2), strides=2)(X1)  # l/4,b/4,128
    X2 = L.Conv2D(256, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.LeakyReLU(alpha=0.2)(X2)
    X2 = L.Conv2D(256, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.LeakyReLU(alpha=0.2)(X2)  # l/4,b/4,256

    X3 = L.MaxPool2D((2, 2), strides=2)(X2)  # l/8,b/8,256
    X3 = L.Conv2D(512, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.LeakyReLU(alpha=0.2)(X3)
    X3 = L.Conv2D(512, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.LeakyReLU(alpha=0.2)(X3)  # l/8,b/8,512

    X4 = L.MaxPool2D((2, 2), strides=2)(X3)  # l/16,b/16,512
    X4 = L.Conv2D(1024, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.LeakyReLU(alpha=0.2)(X4)
    X4 = L.Conv2D(1024, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.LeakyReLU(alpha=0.2)(X4)  # l/16,b/16,1024

    X4 = L.Conv2DTranspose(512, (2, 2), strides=2)(X4)  # l/8,b/8,512
    X4 = L.Concatenate()([X3, X4])  # l/8,b/8,1024
    X4 = L.Conv2D(512, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.Activation('relu')(X4)
    X4 = L.Conv2D(512, (3, 3), padding='same')(X4)
    X4 = L.BatchNormalization()(X4)
    X4 = L.Activation('relu')(X4)  # l/8,b/8,512

    X3 = L.Conv2DTranspose(256, (2, 2), strides=2)(X4)  # l/4,b.4,256
    X3 = L.Concatenate()([X2, X3])  # l/4,b/4,512
    X3 = L.Conv2D(256, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.Activation('relu')(X3)
    X3 = L.Conv2D(256, (3, 3), padding='same')(X3)
    X3 = L.BatchNormalization()(X3)
    X3 = L.Activation('relu')(X3)  # l/4,b/4,256

    X2 = L.Conv2DTranspose(128, (2, 2), strides=2)(X3)  # l/2,b/2,128
    X2 = L.Concatenate()([X1, X2])  # l/2,b/2,256
    X2 = L.Conv2D(128, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.Activation('relu')(X2)
    X2 = L.Conv2D(128, (3, 3), padding='same')(X2)
    X2 = L.BatchNormalization()(X2)
    X2 = L.Activation('relu')(X2)  # l/2,b/2,128

    X1 = L.Conv2DTranspose(64, (2, 2), strides=2)(X2)  # l,b,64
    X1 = L.Concatenate()([X0, X1])  # l,b,128
    X1 = L.Conv2D(64, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.Activation('relu')(X1)
    X1 = L.Conv2D(64, (3, 3), padding='same')(X1)
    X1 = L.BatchNormalization()(X1)
    X1 = L.Activation('relu')(X1)  # l,b,64

    X0 = L.Conv2D(3, (1, 1), strides=1)(X1)  # l,b,3
    model = tensorflow.keras.Model(inputs=Xinpt, outputs=X0)
    return model

def train_and_predict():
    print('-' * 40)
    print('Loading and preprocessing train data...')
    print('-' * 40)
    imgs_train, imgs_mask_train = load_train_data()
    validation_x, validation_y = load_val_data()

    imgs_mask_train = imgs_mask_train.astype(np.float32)
    validation_y = validation_y.astype(np.float32)

    # imgs_train, imgs_mask_train = augmentation(imgs_train, imgs_mask_train)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    # model = models.att_unet_2d(input_size=(256, 256, 3),
    #                         filter_num=[32, 64, 128, 256, 512],
    #                        n_labels=1,
    #                        stack_num_down=2, stack_num_up=2,
    #                        activation='ReLU',
    #                        atten_activation='ReLU', attention='add',
    #                        output_activation='Sigmoid',
    #                        batch_norm=True, pool=True, unpool=True,
    #                        backbone="ResNet50", weights="imagenet",
    #                        freeze_backbone=False, freeze_batch_norm=False,
    #                        name='attunet')

    # model = models.resunet_a_2d(input_size=(256, 256, 1), filter_num=[32, 64, 128, 256, 512], dilation_num=[1, 3, 15, 31],
    #                             n_labels=1, activation="ReLU", output_activation="Sigmoid")

    # model = models.unet_3plus_2d(input_size=(256, 256, 3),
    #                         filter_num_down=[32, 64, 128, 256, 512],
    #                         stack_num_down=2, stack_num_up=2,
    #                         n_labels=1, deep_supervision=False, weights="imagenet",
    #                         freeze_backbone=False, batch_norm=True, freeze_batch_norm=False)

    # plt.imshow(imgs_train[0])
    # plt.show()
    # plt.imshow(imgs_mask_train[0], cmap="gray")
    # plt.show()
    # exit(0)

    # model = models.unet_plus_2d(input_size=(256, 256, 3),
    #                           filter_num=[32, 64, 128, 256, 512],
    #                          n_labels=255, backbone="ResNet101", weights="imagenet",
    #                            freeze_backbone=False, freeze_batch_norm=False)

    # print(imgs_train.shape)
    # model = models.unet_2d(input_size=(128, 128, 3),
    #                           filter_num=[32, 64, 128, 256, 512],
    #                          n_labels=255,
    #                         backbone="ResNet101", weights="imagenet",
    #                         freeze_backbone=False, freeze_batch_norm=False)

    # model = models.r2_unet_2d(input_size=(256, 256, 3),
    #                           filter_num=[32, 64, 128, 256, 512],
    #                          n_labels=1)

    model = UNet()

    # Compile the model
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=1e-2), loss=mean_squared_error, metrics=[mean_squared_error])

    model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=10, verbose=1, shuffle=True,
              validation_data=(validation_x, validation_y),
              callbacks=[tensorboard])

    model.save_weights('weights.h5')

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'resss'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(pred_dir, image_id + '.JPEG'), image)


if __name__ == '__main__':
    train_and_predict()
