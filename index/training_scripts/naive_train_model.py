# --------------------------------------------------------------------
# naive_train_model.py - standalone script for training a single naive
# learned index on one iteration of the integer training data.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import sys
import time

import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 2000

if len(sys.argv) != 2:
    raise Exception("This program requires an input argument (filename)")

filename = sys.argv[1]

column_names = ['key', 'index']

raw_dataset = pd.read_csv(filename, names=column_names,
                          na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)
train_dataset = raw_dataset.copy()
train_labels = train_dataset.pop('index')
# Normalise data
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_dataset = (train_dataset - train_stats['mean']) / train_stats['std']

model = keras.Sequential([
    layers.Dense(32, activation=tf.nn.relu, input_shape=(1,)),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(1)
])

optimizer = tf.train.RMSPropOptimizer(0.01)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])
max_epochs = EPOCHS


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


callbacks = [keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                           min_delta=0, patience=100,
                                           verbose=0, mode='auto',
                                           baseline=None),
             keras.callbacks.ModelCheckpoint(
                 "weights_{}.h5".format(int(time.time())),
                 monitor='mean_absolute_error',
                 verbose=0, save_best_only=True,
                 save_weights_only=True,
                 mode='auto',
                 period=1),
             PrintDot()]
history = model.fit(train_dataset, train_labels, epochs=max_epochs,
              validation_split=1,
          verbose=0, callbacks=callbacks)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
epochs = len(hist['epoch'])
plt.plot(hist['epoch'][epochs-21:epochs-1],
                     hist['mean_absolute_error'][epochs-21:epochs-1],
                     label='Train Error')
plt.legend()

plt.savefig("history_{}".format(int(time.time())))
plt.close()
