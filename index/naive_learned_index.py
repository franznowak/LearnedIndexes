# --------------------------------------------------------------------
# naive_learned_index - Model class implementing a naive learned index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import layers
from custom_exceptions import ModelNotTrainedException


class Model:
    """
    A neural network with one-dimensional input and output and definable
    complexity.
    Usage: initialise, then train, then predict.

    """
    def __init__(self, complexity, training_data, step_size=0.001):
        """
        Initialises the neural network.

        :param complexity: a list of widths of the hidden layers of the neural
        network
        :param training_data: an array containing key, index pairs
        :param step_size: the alpha value for training the neural net

        """
        self.complexity = complexity
        self.model = self._build_model(complexity, step_size=step_size)

        self.stats = None
        self.data, self.labels = self._prepare_data(training_data)
        self.history = None
        self.trained = False

    def train(self, weights_path, checkpoint_name, epochs=100):
        """
        Trains the model on the training_data and saves the weights.

        :param weights_path: path where model is to be stored
        :param checkpoint_name: file in which to store model weights
        :param epochs: number of epochs to train for, default:100

        :return: history object of the training process

        """
        max_epochs = epochs

        if not os.path.isdir(weights_path):
            os.mkdir(weights_path)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                          min_delta=0, patience=0,
                                          verbose=0, mode='auto',
                                          baseline=None),
            keras.callbacks.ModelCheckpoint(weights_path + checkpoint_name,
                                            monitor='mean_absolute_error',
                                            verbose=0, save_best_only=True,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=1)]

        history = self.model.fit(
            self.data, self.labels,
            epochs=max_epochs, validation_split=1, verbose=0,
            callbacks=callbacks)

        self.trained = True
        self.history = history

        return history

    def load_weights(self, weights_filename):
        """
        Loads the weights from a file

        :param weights_filename: name of the weights file

        """
        self.model.load_weights(weights_filename)
        self.trained = True

    def predict(self, key):
        """
        Predicts the position of the data based on the key

        :param key: int key input to the model
        :return: prediction of position

        """
        if not self.trained:
            raise ModelNotTrainedException()

        normed_key = self._norm(key)
        return int(self.model.predict([normed_key]).flatten())

    def plot_history(self, output_path='', output_filename='history.png',
                     metric='mae'):
        """
        Saves a plot of the training history to a png file.

        :param output_filename: filename for the history picture
        :param output_path: path where history shall be stored
        :param metric: either 'mse' or 'mae' for mean squared error or mean
        absolute error, respectively

        """
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch

        if metric == 'mae':
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Abs Error')
            epochs = len(hist['epoch'])
            plt.plot(hist['epoch'][epochs-21:epochs-1],
                     hist['mean_absolute_error'][epochs-21:epochs-1],
                     label='Train Error')
            plt.legend()
        elif metric == 'mse':
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Square Error')
            plt.plot(hist['epoch'], hist['mean_squared_error'],
                     label='Train Error')
            plt.legend()
        else:
            raise NameError("metric \"{}\" not recognised".format(metric))

        if output_path != '' and not os.path.isdir(output_path):
            os.mkdir(output_path)
        plt.savefig(output_path + output_filename)
        plt.close()

    def _prepare_data(self, dataset):
        """
        Prepares data from csv file for training.

        :param dataset: a pandas dataframe of key, index pairs

        :return: tuple of training_data (key,label pairs), training_labels

        """
        train_dataset = dataset.copy()
        train_labels = train_dataset.pop('index')

        # Normalise data
        self.stats = train_dataset.astype('int64').describe().transpose()

        train_dataset = self._norm(train_dataset)

        return train_dataset, train_labels

    def _get_testing_data(self, path):
        """
        Prepares data for testing the model.

        :param path: the path to a csv file with key, index pairs

        :return: normed keys and labels
        """
        dataset = self.load_training_data(path)

        dataset.drop_duplicates(subset="key", keep='first', inplace=True)
        true_labels = dataset.pop('index')

        keys = pd.DataFrame(np.array(
            [i for i in range(len(true_labels.keys()))]))

        normed_keys = self._norm(keys)

        return normed_keys, true_labels

    @staticmethod
    def load_training_data(filename):
        """
        Loads training data from csv file.

        :param filename: the path to a csv file with key, index pairs

        :return: a data frame of keys and values

        """
        column_names = ['key', 'index']
        dataset = pd.read_csv(filename, names=column_names,
                              na_values="?", comment='\t',
                              sep=",", skipinitialspace=True)

        return dataset

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
                model_layers.append(layers.Dense(complexity[0],
                                    activation=tf.nn.relu,
                                    input_shape=[1]))
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

    def _norm(self, x):
        """
        Normalises data for training with a neural net using the training data.

        :param x: the dataset
        :return: normalised data for training

        """
        return (x - self.stats['mean']) / self.stats['std']
