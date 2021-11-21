import os
from cv2 import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob

#验证 调用训练好的模型 未写完
def load_characters():
    charset = []
    if os.path.exists('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt'):
        with open('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt', 'r', encoding="utf-8") as f:
            charset = f.readlines()
            charset = [i.strip() for i in charset]
    return charset


target_size = 64
characters = load_characters()
num_classes = len(characters)
# use_keras_fit = False
use_keras_fit = True


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    # original is 64x64, add a channel dim
    x['image'] = tf.expand_dims(x['image'], axis=-1)
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] = (x['image'] - 128.) / 128.
    return x['image'], x['label']


def get_model():
    # init model
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(1042, activation="relu"),
        keras.layers.Dense(3765, activation="softmax")
    ])
    model.build((64,64,1))
    model.load_weights('D:/WorkSpace/PycharmProjects/ML_2021/checkpoints/cn_ocr.h5')
    return model


def predict(model, img_f):
    ori_img = cv2.imread(img_f)
    img = (ori_img - 128.)/128.
    print(img.shape)
    out = model(img).numpy()
    print('predict: {}'.format(characters[np.argmax(out[0])]))
    cv2.imwrite('assets/pred_{}.png'.format(characters[np.argmax(out[0])]), ori_img)


if __name__ == '__main__':
    img_files = glob.glob('assets/*.png')
    model = get_model()
    for img_f in img_files:
        a = cv2.imread(img_f)
        cv2.imshow('rr', a)
        predict(model, img_f)
        cv2.waitKey(0)
