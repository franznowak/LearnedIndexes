from __future__ import absolute_import, division, print_function

import pathlib
import numpy as np

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    dataset_path = '../training_data/run0inter3'

    column_names = ['key', 'index']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=",", skipinitialspace=True)

    dataset = raw_dataset.copy()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # sns.pairplot(train_dataset[["index", "key"]], diag_kind="kde")

    train_labels = train_dataset.pop('index')
    test_labels = test_dataset.pop('index')

    # Normalise data
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    model = build_model(train_dataset)

    print(model.summary())

    EPOCHS = 1000  # Change depending on convergence, and max

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()]) # Different callback


def norm(x, stats):
    return (x - stats['mean']) / stats['std']


def build_model(train_dataset):
    model = keras.Sequential([
      layers.Dense(32, activation=tf.nn.relu, input_shape=[len(
         train_dataset.keys())]),
      layers.Dense(32, activation=tf.nn.relu),
      layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs):
     if epoch % 100 == 0: print('')
     print('.', end='')


if __name__ == "__main__":
    main()
