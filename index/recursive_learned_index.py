import time

from index.naive_learned_index import Model
import pandas as pd
import config


EPOCHS = 2000
DATA_SET = "Integers_1x10x1k/"
WEIGHTS_PATH = "../data/indexes/recursive_learned_index/" + DATA_SET
DATA_PATH = "../data/datasets/" + DATA_SET
GRAPH_PATH = "../data/graphs/recursive_learned_index/" + DATA_SET


def train_recursive(stages, all_data, nn_complexity, step_size):
    """
    Hybrid end-to-end training algorithm as described in The Case For
    Learned Index Structures, Kraska et al. p8.

    :param stages:
        list of widths of the regression tree for each level.
    :param all_data:
        dataframe of all the data (key-index pairs) for the dataset.
    :param  nn_complexity:
        list of shapes of the NNs for all NN-model in the regression tree.
        each shape is a list of widths of neurons for each layer.
    :param step_size:
        step value for the gradient descent of the NN-training.

    :return trained_index

    """
    trained_index = []

    n_stages = len(stages)

    tmp_records = []
    for i in range(n_stages):
        tmp_records.append([])
        for j in range(stages[i]):
            df = pd.DataFrame([], columns=['key', 'index'])
            tmp_records[i].append(df)

    tmp_records[0][0] = all_data
    data_size = len(all_data.index)

    for i in range(n_stages):
        trained_index.append([])
        for j in range(stages[i]):
            model = Model(nn_complexity[i],
                          tmp_records[i][j],
                          "{}weights{}_{}.h5"
                          .format(WEIGHTS_PATH, i, j, step_size))
            model.train(EPOCHS)
            model.plot_history('{}{}_history_{}_{}.png'
                               .format(GRAPH_PATH, int(time.time()), i, j))
            trained_index[i].append(model)
            if i < n_stages-1:
                rs = tmp_records[i][j]

                for k in rs.index:
                    r = rs.loc[k, :]
                    p = int(trained_index[i][j].predict(rs.at[k, 'key'])
                            * (stages[i+1]) / data_size)
                    p = min(stages[i+1]-1, max(0, p))
                    tmp_records[i+1][p] = tmp_records[i+1][p].append(
                        r.astype('int64'))

            print("model[{}][{}]:{} entries"
                  .format(i, j, len(tmp_records[i][j].index)))

    return trained_index


if __name__ == "__main__":
    train_recursive([1, 2, 4], Model.load_training_data(
        '{}run{}inter{}'.format(DATA_PATH, 0, 9)),
                    [[4, 4], [4, 4], [1]],
                    config.STEP_SIZE)
