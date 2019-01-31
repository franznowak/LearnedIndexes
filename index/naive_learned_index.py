from __future__ import absolute_import, division, print_function

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


def main():
    histories = train_model(1, 2, 10)
    for history in histories:
        plot_history(history)


def train_model(runs, interpolations, epochs):
    histories = []
    for i in range(runs):
        for j in range(interpolations):
            dataset_path = '../data/run{}inter{}'.format(i,j)

            training_data, training_labels = prepare_data(dataset_path)
            size = len(training_data.keys())

            model = build_model(size)

            h = train_upto(model, training_data, training_labels,
                           'models/NN_run{}inter{}.h5'.format(i,j), epochs)
            histories.append(h)
    return histories


def predict(run, interpolation):
    dataset_path = '../data/run{}inter{}'.format(run, interpolation)

    training_data, training_labels = prepare_data(dataset_path)
    size = len(training_data.keys())

    model = build_model(size)

    model.load_weights('models/NN_run{}inter{}.h5'.format(run, interpolation))

    # We are over-fitting, so test using training data
    test_data = training_data
    test_labels = training_labels

    test_predictions = model.predict(test_data).flatten()
    plot_prediction(test_labels, test_predictions)


def train_upto(model, data, labels, checkpoint_name, epochs = 100):
    max_epochs = epochs
    callbacks = [keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                               min_delta=0, patience=10,
                                               verbose=0, mode='auto',
                                               baseline=None),
                 keras.callbacks.ModelCheckpoint(checkpoint_name,
                                                 monitor='mean_absolute_error',
                                                 verbose=0, save_best_only=True,
                                                 save_weights_only=True,
                                                 mode='auto',
                                                 period=1), PrintDot()]

    history = model.fit(
        data, labels,
        epochs=max_epochs, validation_split=1, verbose=0,
        callbacks=callbacks)

    return history


def norm(x, stats):
    return (x - stats['mean']) / stats['std']


def prepare_data(path):
    column_names = ['key', 'index']
    raw_dataset = pd.read_csv(path, names=column_names,
                              na_values="?", comment='\t',
                              sep=",", skipinitialspace=True)

    train_dataset = raw_dataset.copy()

    train_labels = train_dataset.pop('index')

    # Normalise data
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    train_dataset = norm(train_dataset, train_stats)

    return train_dataset, train_labels


def build_model(input_size):
    model = keras.Sequential([
      layers.Dense(32, activation=tf.nn.relu, input_shape=[input_size]),
      layers.Dense(32, activation=tf.nn.relu),
      layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def plot_history(history, metric='mae'):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    if metric == 'mae':
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.legend()
        plt.ylim([0, 1000])
    elif metric == 'mse':
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')
        plt.legend()
        plt.ylim([0, 800])
    else:
        raise NameError("metric \"{}\" not recognised".format(metric))

    plt.show()


def plot_prediction(labels, prediction):
    plt.scatter(labels, prediction)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([0, len(labels)], [0, len(labels)])
    plt.show()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


if __name__ == "__main__":
    main()

