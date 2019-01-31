from __future__ import absolute_import, division, print_function

import sys

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 100

if len(sys.argv) != 3:
    raise Exception("This program requires 2 input arguments")

run = int(sys.argv[1])
inter = int(sys.argv[2])

filename = "data/run{}inter{}".format(run, inter)

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

size = len(train_dataset.keys())

model = keras.Sequential([
    layers.Dense(32, activation=tf.nn.relu, input_shape=[size]),
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
             keras.callbacks.ModelCheckpoint('models/weights{}_{}.h5'.format(
                                                                run, inter),
                                             monitor='mean_absolute_error',
                                             verbose=0, save_best_only=True,
                                             save_weights_only=True,
                                             mode='auto',
                                             period=1)]
model.fit(train_dataset, train_labels, epochs=max_epochs, validation_split=1,
          verbose=0, callbacks=callbacks)
