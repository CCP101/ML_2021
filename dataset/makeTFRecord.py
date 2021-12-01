import tensorflow as tf
import os
import time
from cv2 import cv2
import numpy as np

if __name__ == '__main__':
    tfrecord_f = './TFrecord/train.tfrecord'
    # 数据集里为了简便及计算方便，不存储对应中文，使用序号表示
    # charset = []
    # if os.path.exists('characters.txt'):
    #     with open('characters.txt', 'r', encoding="utf-8") as f:
    #         charset = f.readlines()
    #         charset = [i.strip() for i in charset]
    g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\train_final_shuffle\\")
    # TFRecord读写器
    with tf.io.TFRecordWriter(tfrecord_f) as tfrecord_writer:
        for path, dir_list, file_list in g:
            for file_name in file_list:
                img_path = path + "\\" + file_name
                img = cv2.imread(img_path)
                # 图片由3通道彩色图转换为1通道灰度图
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 转为numpy数组，比特方式存入TFrecord
                img = np.array(img)
                # 获取图片对应的标签
                label = file_name.split("_")[1]
                label = int(label.split(".")[0])
                # 创建TFrecord制作规则
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[64])),
                        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[64]))
                    }))
                tfrecord_writer.write(example.SerializeToString())
                # 每张图片存入label、image、width、height四种属性
                print('add {} examples. {}'.format(label, img_path))
