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
    img = tf.cast(tf.reshape(img, [w, h, 1]), dtype=tf.float32)
    return {'image': img, 'label': label, 'width': w, 'height': h}


def load_ds():
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\TFrecord\\train.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example)
    return ds


def load_val_ds():
    input_files = ['D:\\WorkSpace\\PycharmProjects\\ML_2021\\dataset\\TFrecord\\test.tfrecord']
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
    train_dataset = train_dataset.map(preprocess)
    train_dataset = train_dataset.shuffle(640)
    train_dataset = train_dataset.batch(64)
    train_dataset = train_dataset.repeat()
    val_ds = load_val_ds()
    val_ds = val_ds.map(preprocess)
    val_ds = val_ds.shuffle(640)
    val_ds = val_ds.batch(64)
    val_ds = val_ds.repeat()

    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding="valid", strides=(1, 1), data_format="channels_last", input_shape=(64, 64, 1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, (3, 3), padding="valid", strides=(1, 1), data_format="channels_last", activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3765, activation='softmax')
    ])
    start_epoch = 0
    ckpt_path = "D:\\WorkSpace\\PycharmProjects\\ML_2021\\checkpoints\\"
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    model.save_weights(ckpt_path.format(epoch=0))
    model.save(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        print('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        print('passing resume since weights not there. training from scratch')
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1, period=100),
                 tf.keras.callbacks.TensorBoard(run_logdir)]
    learning_rate = 0.01
    decay = 1e-6
    momentum = 0.8
    nesterov = True
    sgd_optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=decay,
                                            momentum=momentum, nesterov=nesterov)
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
        model.save_weights(ckpt_path.format(epoch=0))
        print('keras model saved.')
