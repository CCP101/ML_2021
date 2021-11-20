import tensorflow as tf
import os
import time
from cv2 import cv2
import numpy as np


if __name__ == '__main__':
    tfrecord_f = './TFrecord/train.tfrecord'
    # charset = []
    # if os.path.exists('characters.txt'):
    #     with open('characters.txt', 'r', encoding="utf-8") as f:
    #         charset = f.readlines()
    #         charset = [i.strip() for i in charset]
    g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\train\\")
    with tf.io.TFRecordWriter(tfrecord_f) as tfrecord_writer:
        for path, dir_list, file_list in g:
            for file_name in file_list:
                img_path = path + "\\" + file_name
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img)
                label = int(file_name.split("_")[0])
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[64])),
                        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[64]))
                    }))
                tfrecord_writer.write(example.SerializeToString())
                print('add {} examples. {}'.format(label, img_path))
