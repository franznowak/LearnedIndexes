# --------------------------------------------------------------------
# hybrid_learned_index.py - function outlining training of a hybrid learned
# index.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

from .naive_learned_index import Model


EPOCHS = 100
WEIGHTS_PATH = "../data/indexes/recursive_learned_index"
DATA_PATH = "../data/datasets"


def train_recursive(stages, all_data, nn_complexity, threshold:int):
    """
    Hybrid end-to-end training algorithm as described in The Case For
    Learned Index Structures, Kraska et al. p8.

    :param stages:
        list of widths of the regression tree for each level.
    :param all_data:
        list of all the data (key-index pairs) for the dataset.
    :param  nn_complexity:
        list of shapes of the NNs for all NN-model in the regression tree.
        each shape is a list of widths of neurons for each layer.
    :param threshold:
        max absolute error of NN before other index is used.

    :return trained_index

    """
    trained_index = []

    M = len(stages)

    tmp_records = []
    for i in range(M):
        tmp_records.append([])
        for j in range(stages[i]):
            tmp_records[i].append([])

    tmp_records[0][0] = all_data

    for i in range(M):
        trained_index.append([])
        for j in range(stages[i]):
            model = Model(nn_complexity[i][j], tmp_records[i][j], "{}weights{}"
                                                                  "_{}.h5"
                          .format(WEIGHTS_PATH, i, j))
            model.train()
            trained_index[i].append(model)

            if i < M:
                for r in tmp_records[i][j]:
                    p = int(trained_index[i][j].predict(r.key)/stages[i+1])
                    tmp_records[i+1][p].add(r)

    # TODO: if error too high (above threshold) replace with b-tree here

    return trained_index
