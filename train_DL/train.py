import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import logging


def parse_example(record):
    #读取TFrecord文件，与制作方式对应读取
    dict_data = tf.io.parse_single_example(record, features={
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64)})
    label = tf.cast(dict_data['label'], tf.int64)
    w = tf.cast(dict_data['width'], tf.int64)
    h = tf.cast(dict_data['height'], tf.int64)
    #对应numpy编码方式读取
    img = tf.io.decode_raw(dict_data['image'], tf.uint8)
    #重新转换为图片
    img = tf.cast(tf.reshape(img, [w, h, 1]), dtype=tf.float32)
    return {'image': img, 'label': label, 'width': w, 'height': h}


def load_ds():
    #读取训练TFrecord
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\TFrecord\\train.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example)
    return ds


def load_val_ds():
    #读取验证TFrecord
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\TFrecord\\test.tfrecord']
    ds2 = tf.data.TFRecordDataset(input_files)
    ds2 = ds2.map(parse_example)
    return ds2


def load_characters():
    #读取字符集
    charset = []
    if os.path.exists('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt'):
        with open('D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\characters.txt', 'r', encoding="utf-8") as f:
            charset = f.readlines()
            charset = [i.strip() for i in charset]
    return charset


def preprocess(x):
    #归一化操作
    x['image'] = (x['image'] - 128.) / 128.
    return x['image'], x['label']


def get_run_logdir():
    #创建日志文件夹
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


if __name__ == '__main__':
    #日志记录配置
    root_logdir = os.path.join(os.curdir, "my_logs")
    run_logdir = get_run_logdir()
    #字符集读取
    all_characters = load_characters()
    num_classes = len(all_characters)
    print('all characters: {}'.format(num_classes))
    #数据集设置
    train_dataset = load_ds()
    train_dataset = train_dataset.map(preprocess)
    #TODO:shuffle需要手动实现 这里有问题
    train_dataset = train_dataset.shuffle(buffer_size=3000, seed=42)
    train_dataset = train_dataset.batch(200)
    train_dataset = train_dataset.repeat()
    val_ds = load_val_ds()
    val_ds = val_ds.map(preprocess)
    val_ds = val_ds.shuffle(buffer_size=3000, seed=42)
    val_ds = val_ds.batch(200)
    val_ds = val_ds.repeat()
    #模型搭建 Conv2D卷积 MaxPool2D池化 Dense全连接
    model = keras.models.Sequential([
        keras.layers.Conv2D(input_shape=(64, 64, 1), filters=32, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(1000, activation='softmax')
        # keras.layers.Conv2D(96, (11, 11), strides=(3, 3), input_shape=(64, 64, 1), activation="relu"),
        # keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # keras.layers.Conv2D(256, (5, 5), padding="same", strides=(1, 1), activation="relu"),
        # keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # keras.layers.Conv2D(384, (3, 3), padding="same", strides=(1, 1), activation="relu"),
        # keras.layers.Conv2D(384, (3, 3), padding="same", strides=(1, 1), activation="relu"),
        # keras.layers.Conv2D(256, (3, 3), padding="same", strides=(1, 1), activation="relu"),
        # keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # keras.layers.Flatten(),
        # keras.layers.Dense(4096, activation='relu'),
        # keras.layers.Dropout(0.25),
        # keras.layers.Dense(4096, activation='relu'),
        # keras.layers.Dropout(0.25),
        # keras.layers.Dense(1000, activation='softmax')
    ])
    start_epoch = 0
    ckpt_path = "D:\\WorkSpace\\PycharmProjects\\ML_2021\\checkpoints\\"
    #回调配置
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1, period=100),
                tf.keras.callbacks.TensorBoard(run_logdir)]
    #SGD随机梯度下降配置
    learning_rate = 0.01
    decay = 1e-6
    momentum = 0.8
    nesterov = True
    sgd_optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=decay,
                                            momentum=momentum, nesterov=nesterov)
    #因为是one-hot标签，所以采用离散交叉熵损失函数
    model.compile(
        optimizer=sgd_optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'])
    model.summary()
    try:
        tf.data.experimental.ignore_errors()
        model.fit(
            train_dataset,
            validation_data=val_ds,
            validation_steps=100,
            epochs=15000,
            steps_per_epoch=1024,
            callbacks=callbacks)
    except KeyboardInterrupt:
        #当CTRL+C时，停止训练，保存模型
        model.save(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))
        model.save_weights(ckpt_path.format(epoch=200))
        print('keras model saved.')
