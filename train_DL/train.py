import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import logging


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
    x['image'] = (x['image']) / 255.
    return x['image'], x['label']


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


if __name__ == '__main__':
    root_logdir = os.path.join(os.curdir, "my_logs")
    run_logdir = get_run_logdir()
    all_characters = load_characters()
    num_classes = len(all_characters)
    print('all characters: {}'.format(num_classes))
    train_dataset = load_ds()
    train_dataset = train_dataset.shuffle(640).map(preprocess).batch(64).repeat()
    val_ds = load_val_ds()
    val_ds = val_ds.shuffle(640).map(preprocess).batch(64).repeat()
    # for data in val_ds.take(2):
    #     print(data)
    model = keras.models.Sequential([
        keras.layers.Conv2D(input_shape=(64, 64, 3), filters=32, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        keras.layers.Dropout(0.25),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(3756, activation='softmax')
        # keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        # keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
        # keras.layers.MaxPool2D(),
        # keras.layers.Flatten(),
        # keras.layers.Dropout(0.25),
        # keras.layers.Dense(128, activation="relu"),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(3756, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])
    start_epoch = 0
    ckpt_path = "D:\\WorkSpace\\PycharmProjects\\ML_2021\\checkpoints\\"
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        print('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        print('passing resume since weights not there. training from scratch')
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1, period=500),
                 tf.keras.callbacks.TensorBoard(run_logdir)]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    try:
        tf.data.experimental.ignore_errors()
        model.fit(
            train_dataset,
            validation_data=val_ds,
            validation_steps=1000,
            epochs=15000,
            steps_per_epoch=1024,
            callbacks=callbacks)
    except KeyboardInterrupt:
        model.save_weights(ckpt_path.format(epoch=0))
        print('keras model saved.')
    model.save_weights(ckpt_path.format(epoch=0))
    model.save(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))
