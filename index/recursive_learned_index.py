# --------------------------------------------------------------------
# recursive_learned_index.py - class for making predictions using index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import time

from index.naive_learned_index import Model
import pandas as pd


class RecursiveLearnedIndex:
    """
    A regression tree of neural networks that takes a one dimensional input
    and returns a prediction of the position in the dataset by iteratively
    predicting and thus finding the most accurate model.

    """
    def __init__(self, stages, nn_complexity):
        """
        Initialises the multilevel learned index.

        :param stages:
            list of widths of the regression tree for each level.
        :param  nn_complexity:
            list of shapes of the NNs for all NN-model in the regression tree.
            each shape is a list of widths of neurons for each layer.

        :return trained_index

        """
        self.stages = stages
        self.nn_complexity = nn_complexity
        self.data_size = 0  # the total number of true indexes
        self.index = None

    def train(self, all_data, weights_path='', graph_path='', epochs=100):
        """
        Hybrid end-to-end training algorithm as described in The Case For
        Learned Index Structures, Kraska et al.

        :param all_data:
            dataframe of all the data (key-index pairs) of the dataset.
        :param weights_path:
            path where model weights shall be stored.
        :param graph_path:
            path where training history graphs shall be stored.
        :param epochs:
            number of epochs for the gradient descent of the NN-training.

        :return trained_index

        """
        trained_index = []

        n_stages = len(self.stages)
        self.data_size = len(all_data.index)

        tmp_records = []
        for i in range(n_stages):
            tmp_records.append([])
            for j in range(self.stages[i]):
                df = pd.DataFrame([], columns=['key', 'index'])
                tmp_records[i].append(df)

        tmp_records[0][0] = all_data

        for i in range(n_stages):
            trained_index.append([])
            for j in range(self.stages[i]):
                model = Model(self.nn_complexity[i],
                              tmp_records[i][j])
                model.train("{}weights{}_{}.h5"
                            .format(weights_path, i, j), epochs)
                model.plot_history('{}{}_history_{}_{}.png'
                                   .format(graph_path, int(time.time()), i, j))
                trained_index[i].append(model)
                if i < n_stages-1:
                    rs = tmp_records[i][j]

                    for k in rs.index:
                        r = rs.loc[k, :]
                        p = trained_index[i][j].predict(rs.at[k, 'key'])
                        q = self._get_next_model_index(i, p)
                        tmp_records[i+1][q] = tmp_records[i+1][q].append(
                            r.astype('int64'))

                print("model[{}][{}]:{} entries"
                      .format(i, j, len(tmp_records[i][j].index)))

        self.index = trained_index

    def load_models(self, weights_path, data):
        """
        Loads previously trained model weights from a specified location and
        initialises the index.

        :param weights_path: the location containing the model weights
        :param data: dataframe of all the data (key-index pairs) of the
        dataset.

        """
        self.data_size = len(data.index)

        self.index = []
        for i in range(len(self.stages)):
            self.index.append([])
            for j in range(self.stages[i]):
                model = Model(self.nn_complexity[i], data)
                model.load_weights("{}weights{}_{}.h5"
                                   .format(weights_path, i, j))
                self.index[i].append(model)

    def _get_next_model_index(self, current_stage, current_prediction):
        """
        Returns the index of the next model in the next stage of the tree to
        be used for prediction.

        :param current_stage: the current stage at which predictions are made
        :param current_prediction: the last prediction made by the current model
        :return: index of the next model down the tree

        """
        p = int(current_prediction * self.stages[current_stage + 1] /
                self.data_size)
        return min(self.stages[current_stage + 1] - 1, max(0, p))

    def predict(self, key):
        """
        Predicts the position of the data based on the key.

        :param key: int key input to the model
        :return: prediction of position

        """
        j = 0
        n_stages = len(self.stages)
        for i in range(n_stages - 1):
            j = self._get_next_model_index(i, self.index[i][j].predict(key))

        return self.index[n_stages-1][j].predict(key)
