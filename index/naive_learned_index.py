from __future__ import absolute_import, division, print_function

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


def main():
    dataset_path = '../training_data/run0inter3'

    column_names = ['key', 'index']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=",", skipinitialspace=True)

    train_dataset = raw_dataset.copy()

    train_labels = train_dataset.pop('index')
    test_labels = train_labels

    # Normalise data
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = normed_train_data

    model = build_model(train_dataset)

    max_epochs = 1000 # Change depending on convergence, and max
    callbacks = [keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                               min_delta=0, patience=2,
                                               verbose=0, mode='auto',
                                               baseline=None), PrintDot()]

    history = model.fit(
        normed_train_data, train_labels,
        epochs=max_epochs, validation_split=0.2, verbose=0,
        callbacks=callbacks)

    plot_history(history)

    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()


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


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.legend()
    plt.ylim([0, 50])
    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Square Error')
    # plt.plot(hist['epoch'], hist['mean_squared_error'],
    #          label='Train Error')
    # plt.legend()
    # plt.ylim([0, 800])
    plt.show()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


if __name__ == "__main__":
    main()

