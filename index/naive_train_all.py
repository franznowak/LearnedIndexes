#!/home/fgn24/tensorflow-env/bin/python
from __future__ import absolute_import, division, print_function

import sys

import pandas as pd

import tensorflow as tf
import keras
from keras import layers

EPOCHS = 100

weigths_path = "../data/indexes/naive_learned_index/Integers_100x10x100k/"

for run in range(100):
    for inter in range(10):
        import os.path
        if os.path.isfile("{}weights{}_{}.h5".format(weigths_path, run, inter)):
                continue

        filename = "../data/datasets/Integers_100x10x100k/run{}inter{}"\
            .format(run, inter)
        column_names = ['key', 'index']

        raw_dataset = pd.read_csv(filename, names=column_names,
                                  na_values="?", comment='\t',
                                  sep=",", skipinitialspace=True)
        train_dataset = raw_dataset.copy()
        train_labels = train_dataset.pop('index')
        # Normalise data
        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()
        train_dataset = (train_dataset - train_stats['mean']) / train_stats['std']  # ?

        model = keras.Sequential([
            layers.Dense(32, activation=tf.nn.relu, input_shape=[1]),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        max_epochs = EPOCHS
        callbacks = [keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                                   min_delta=0, patience=10,
                                                   verbose=0, mode='auto',
                                                   baseline=None),
                     keras.callbacks.ModelCheckpoint('{}/weights{}_{}.h5'
                                                     .format(weigths_path, run, inter),
                                                     monitor='mean_absolute_error',
                                                     verbose=0, save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto', period=1)]
        model.fit(train_dataset, train_labels, epochs=max_epochs, validation_split=1,
                  verbose=0, callbacks=callbacks)
