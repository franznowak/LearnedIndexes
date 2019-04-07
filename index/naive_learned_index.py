from __future__ import absolute_import, division, print_function
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import config
import numpy as np


def main():
    config.DATASET_PATH = "../" + config.DATASET_PATH
    config.MODEL_PATH = "../" + config.MODEL_PATH
    # histories = train_model(1, 10, 100)
    # for history in histories:
    #     plot_history(history)
    predict(0, 1)


def train_model(runs, interpolations, epochs):
    histories = []
    for i in range(runs):
        for j in range(interpolations):
            dataset_path = config.DATASET_PATH + 'run{}inter{}'.format(i, j)

            training_data, training_labels = prepare_data(dataset_path)

            model = build_model()

            h = train_upto(model, training_data, training_labels,
                           config.MODEL_PATH+'weights{}_{}.h5'.format(i, j),
                           epochs)
            histories.append(h)
    return histories


def validate(run, interpolation):
    dataset_path = config.DATASET_PATH + 'run{}inter{}'.format(run, interpolation)

    training_data, training_labels = prepare_data(dataset_path)

    model = build_model()
    try:
        model.load_weights(config.MODEL_PATH + 'weights{}_{}.h5'.format(run,
                           interpolation))
    except FileNotFoundError:
        raise FileNotFoundError("No model trained for run{}inter{}".format(
            run, interpolation))
    # We are over-fitting, so test using training data
    test_data = training_data
    test_labels = training_labels

    test_predictions = model.predict(test_data).flatten()

    plot_prediction(test_data, test_predictions)
    return test_labels, test_predictions


def predict(run, interpolation):
    import os.path
    if not os.path.isfile(config.MODEL_PATH
                          + "weights{}_{}.h5".format(run, interpolation)):
        raise FileNotFoundError(
            "No model trained for run{}inter{}".format(run, interpolation));

    dataset_path = config.DATASET_PATH \
                   + 'run{}inter{}'.format(run, interpolation)

    data, labels = get_testing_data(dataset_path)

    model = build_model()

    model.load_weights(config.MODEL_PATH + 'weights{}_{}.h5'.format(run,
                           interpolation))

    prediction = model.predict(data).flatten()

    return labels, prediction


def train_upto(model, data, labels, checkpoint_name, epochs=100):
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
                                                 period=1)]

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


def get_testing_data(path):
    column_names = ['key', 'index']
    dataset = pd.read_csv(path, names=column_names,
                          na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)

    dataset.drop_duplicates(subset="key", keep='first', inplace=True)
    true_labels = dataset.pop('index')

    keys = pd.DataFrame(np.array([i for i in range(len(true_labels.keys()))]))

    # Normalise data
    train_stats = keys.describe()
    train_stats = train_stats.transpose()

    normed_keys = norm(keys, train_stats)

    return normed_keys, true_labels


def build_model():
    model = keras.Sequential([
      layers.Dense(32, activation=tf.nn.relu, input_shape=[1]),
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


if __name__ == "__main__":
    main()


class Model:
    """
    A neural network with one-dimensional input and output and definable
    complexity.
    Usage: initialise, then train, then predict.

    """
    def __init__(self, complexity, training_data, checkpoint_name):
        """
        Initialises the neural network.

        :param complexity: a list of widths of the layers of the neural network
        :param training_data: an array containing key, index pairs
        :param checkpoint_name: file in which to store model weights

        """
        self.complexity = complexity
        self.model = self._build_model(complexity)

        self.training_data = training_data
        self.checkpoint_name = checkpoint_name

        self.trained = False

    def train(self, epochs=100):
        """
        Trains the model on the training_data and saves the weights.

        :return: history object of the training process

        """
        max_epochs = epochs

        (data, labels) = prepare_data(self.training_data)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                          min_delta=0, patience=10,
                                          verbose=0, mode='auto',
                                          baseline=None),
            keras.callbacks.ModelCheckpoint(self.checkpoint_name,
                                            monitor='mean_absolute_error',
                                            verbose=0, save_best_only=True,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=1)]

        history = self.model.fit(
            data, labels,
            epochs=max_epochs, validation_split=1, verbose=0,
            callbacks=callbacks)

        self.trained = True

        return history

    def load(self, weights_filename):
        """
        Loads the weights from a file

        :param weights_filename: name of the weights file

        """
        self.model.load_weights(weights_filename)

    def predict(self, key):
        """
        Predicts the position of the data based on the key

        :param key: int key input to the model
        :return: prediction of position

        """
        return self.model.predict(key).flatten()

    @staticmethod
    def _build_model(complexity, step_size):
        """
        Builds a model according to the specifications given in init.

        :return: a neural network model of specified complexity

        """

        model_layers = []

        for i in range(len(complexity)):
            if i == 0:
                # Input layer takes a single key
                model_layers = layers.Dense(complexity[0],
                                            activation=tf.nn.relu,
                                            input_shape=[1])
            else:
                # Hidden layers
                model_layers.append(layers.Dense(complexity[i],
                                    activation=tf.nn.relu))

        # Output layer outputs single key
        model_layers.append(layers.Dense(1))

        model = keras.Sequential(model_layers)

        optimizer = tf.train.RMSPropOptimizer(step_size)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        return model

    @staticmethod
    def _build_standard_model(step_size):
        """
        Builds a standard model with 1 hidden layer of 32 neurons.

        :return: the model

        """
        model = keras.Sequential([
            layers.Dense(32, activation=tf.nn.relu, input_shape=[1]),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(step_size)

        model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['mae', 'mse'])
