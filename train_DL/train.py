import tensorflow as tf
import numpy as np
import time
import os


def parse_example(record):
    dict_data = tf.io.parse_single_example(record, features={
       'label': tf.io.FixedLenFeature([], tf.int64),
       'image': tf.io.FixedLenFeature([], tf.string),
       'width': tf.io.FixedLenFeature([], tf.int64),
       'height': tf.io.FixedLenFeature([], tf.int64)})
    label = tf.cast(dict_data['label'], tf.int64)
    w = tf.cast(dict_data['width'], tf.int64)
    h = tf.cast(dict_data['height'], tf.int64)
    img = tf.io.decode_raw(dict_data['image'], tf.uint8)
    img = tf.cast(tf.reshape(img, [w, h, 3]), dtype=tf.float32)
    return {'image': img, 'label': label, 'width': w, 'height': h}


def load_ds():
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\train.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example)
    return ds


def load_val_ds():
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\test.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example)
    return ds


def load_characters():
    charset = []
    if os.path.exists('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt'):
        with open('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt', 'r', encoding="utf-8") as f:
            charset = f.readlines()
            charset = [i.strip() for i in charset]
    return charset


def preprocess(x):
    x['image'] = (x['image'] - 128.) / 128.
    return x['image'], x['label']


if __name__ == '__main__':
    all_characters = load_characters()
    num_classes = len(all_characters)
    print('all characters: {}'.format(num_classes))
    train_dataset = load_ds()
    train_dataset = train_dataset.shuffle(100).map(preprocess).batch(16).repeat()
    val_ds = load_val_ds()
    val_ds = val_ds.shuffle(100).map(preprocess).batch(16).repeat()
    for data in val_ds.take(2):
        print(data)
