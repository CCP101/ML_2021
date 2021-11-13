import tensorflow as tf
import os
import time
from cv2 import cv2


if __name__ == '__main__':
    tfrecord_f = './test.tfrecord'
    # charset = []
    # if os.path.exists('characters.txt'):
    #     with open('characters.txt', 'r', encoding="utf-8") as f:
    #         charset = f.readlines()
    #         charset = [i.strip() for i in charset]
    g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\test\\")
    with tf.io.TFRecordWriter(tfrecord_f) as tfrecord_writer:
        for path, dir_list, file_list in g:
            count = 0
            for dir in dir_list:
                g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\test\\" + dir)
                for d_path, d_dir_list, d_file_list in g:
                    for file in d_file_list:
                        img_path = d_path + "\\" + file
                        img = cv2.imread(img_path)
                        w = img.shape[0]
                        h = img.shape[1]
                        label = count
                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                            }))
                        tfrecord_writer.write(example.SerializeToString())
                        print('add {} examples. {}:{}'.format(label, count, file))
                count += 1
