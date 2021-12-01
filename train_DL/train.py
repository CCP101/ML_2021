import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import logging


def parse_example(record):
    """
    读取TFrecord文件，从文件中解压相应内容
    :param record: TFrecord文件
    :return: 返回读取后的字典
    """
    # 与制作方式对应读取
    dict_data = tf.io.parse_single_example(record, features={
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64)})
    # 获取标签
    label = tf.cast(dict_data['label'], tf.int64)
    # 获取宽高
    w = tf.cast(dict_data['width'], tf.int64)
    h = tf.cast(dict_data['height'], tf.int64)
    # 对应numpy编码方式读取
    img = tf.io.decode_raw(dict_data['image'], tf.uint8)
    # 重新转换为图片 已经是单通道灰度图 所以维度为1
    img = tf.cast(tf.reshape(img, [w, h, 1]), dtype=tf.float32)
    return {'image': img, 'label': label, 'width': w, 'height': h}


def load_ds():
    """
    读取训练TFrecord文件，并进行解析
    :return: 训练TFrecord文件解析
    """
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\TFrecord\\train.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example)
    return ds


def load_val_ds():
    """
    读取验证TFrecord文件，并进行解析
    :return: 验证TFrecord文件解析
    """
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\TFrecord\\test.tfrecord']
    ds2 = tf.data.TFRecordDataset(input_files)
    ds2 = ds2.map(parse_example)
    return ds2


def load_characters():
    """
    从文件中读取字符集
    :return: 字符列表
    """
    charset = []
    if os.path.exists('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt'):
        with open('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt', 'r', encoding="utf-8") as f:
            charset = f.readlines()
            charset = [i.strip() for i in charset]
    return charset


def preprocess(x):
    """
    与训练方法一致，将0-255的像素转换为0-1区间
    :param x: TFrecord读取到的对象
    :return: 返回图片与对应标签
    """
    # 归一化操作
    x['image'] = x['image'] / 255.
    return x['image'], x['label']


def get_run_logdir():
    """
    创建日志文件夹
    :return: 日志文件夹路径
    """
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


if __name__ == '__main__':
    # 日志记录配置
    root_logdir = os.path.join(os.curdir, "my_logs")
    run_logdir = get_run_logdir()
    # 字符集读取
    all_characters = load_characters()
    num_classes = len(all_characters)
    print('all characters: {}'.format(num_classes))
    # 训练集读取
    train_dataset = load_ds()
    train_dataset = train_dataset.map(preprocess)
    train_dataset = train_dataset.batch(500)
    train_dataset = train_dataset.repeat()
    # 数据集读取
    val_ds = load_val_ds()
    val_ds = val_ds.map(preprocess)
    val_ds = val_ds.batch(500)
    val_ds = val_ds.repeat()
    # 模型搭建 Conv2D卷积 MaxPool2D池化 Dense全连接 strides步长 padding填充 kernel_size卷积核 pool_size池化核 activation 激活函数
    model = keras.models.Sequential([
        keras.layers.Conv2D(96, kernel_size=(7, 7), strides=(3, 3), input_shape=(64, 64, 1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(256, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(384, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.Conv2D(384, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(1000, activation='softmax')
    ])
    start_epoch = 0
    ckpt_path = "D:\\WorkSpace\\PycharmProjects\\ML_2021\\checkpoints\\"
    # 回调配置
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1, period=100),
                 tf.keras.callbacks.TensorBoard(run_logdir)]
    # SGD随机梯度下降配置
    learning_rate = 0.01
    decay = 1e-6
    momentum = 0.8
    nesterov = True
    sgd_optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=decay,
                                            momentum=momentum, nesterov=nesterov)
    # one-hot标签，采用离散交叉熵损失函数
    model.compile(
        optimizer=sgd_optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'])
    # 打印架构信息
    model.summary()
    try:
        tf.data.experimental.ignore_errors()
        model.fit(
            train_dataset,
            validation_data=val_ds,
            validation_steps=100,
            epochs=15000,
            steps_per_epoch=256,
            callbacks=callbacks)
    except KeyboardInterrupt:
        # 当CTRL+C时，停止训练，保存模型
        model.save(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))
        model.save_weights(ckpt_path.format(epoch=200))
        print('keras model saved.')
