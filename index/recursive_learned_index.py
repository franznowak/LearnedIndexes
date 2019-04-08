from index.naive_learned_index import Model
import config


EPOCHS = 100
WEIGHTS_PATH = "../data/indexes/recursive_learned_index"
DATA_PATH = "../data/datasets/synthetic/Integers_1x10x1k"


def train_recursive(stages, all_data, nn_complexity, step_size):
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
            tmp_records[i].append([])

    tmp_records[0][0] = all_data

    for i in range(n_stages):
        trained_index.append([])
        for j in range(stages[i]):
            model = Model(nn_complexity[i],
                          tmp_records[i][j],
                          "{}weights{}_{}.h5"
                          .format(WEIGHTS_PATH, i, j), step_size)
            model.train()
            trained_index[i].append(model)

            if i < n_stages:
                for r in tmp_records[i][j]:
                    p = int(trained_index[i][j].predict(r.key)/stages[i+1])
                    tmp_records[i+1][p].add(r)

    return trained_index


if __name__ == "__main__":
    train_recursive([1, 2, 3], '{}/run{}inter{}'.format(DATA_PATH, 0, 9),
                    [[4, 4], [4, 4], [1]],
                    config.STEP_SIZE)
