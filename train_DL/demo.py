import os
from cv2 import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob


def load_characters():
    charset = []
    if os.path.exists('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\charactersCut.txt'):
        with open('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\charactersCut.txt', 'r', encoding="utf-8") as f:
            charset = f.readlines()
            charset = [i.strip() for i in charset]
    return charset


def get_model():
    # init model
    model = keras.models.Sequential([
        keras.layers.Conv2D(96, (7, 7), strides=(3, 3), input_shape=(64, 64, 1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(256, (5, 5), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(384, (3, 3), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.Conv2D(384, (3, 3), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.Conv2D(256, (3, 3), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(1000, activation='softmax')
    ])
    model.load_weights('D:/WorkSpace/PycharmProjects/ML_2021/checkpoints/cn_ocr.h5')
    return model


def predict(model, img_f):
    ori_img = cv2.imread(img_f)
    ori_img = cv2.resize(ori_img, (64, 64), interpolation=cv2.INTER_CUBIC)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    # 升到四维空间[batch, height, width, channels]
    ori_img = ori_img / 255.
    ori_img = tf.expand_dims(ori_img, axis=-1)
    img = tf.expand_dims(ori_img, axis=0)
    out = model(img).numpy()
    print('predict: {}:{}'.format(characters[np.argmax(out[0])], img_f))


if __name__ == '__main__':
    characters = load_characters()
    use_keras_fit = True
    img_files = glob.glob('assets/*.png')
    model = get_model()
    for img_f in img_files:
        a = cv2.imread(img_f)
        predict(model, img_f)
