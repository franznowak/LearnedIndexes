import tensorflow as tf
import keras
from keras import layers
import config


EPOCHS = 100
WEIGHTS_PATH = "../data/indexes/multilayer_learned_index"
DATA_PATH = "../data/datasets"


def train_hybrid(stages, all_data, run, inter, NN_complexity, threshold:int):
    """
    Hybrid end-to-end training algorithm as described in The Case For
    Learned Index Structures, Kraska et al. p8.

    :param stages:
        list of widths of the regression tree for each level.
    :param all_data:
        list of all the data (key-index pairs) for the dataset.
    :param NN_complexity:
        list of lists of (width, depth) tuples of the NNs for all NN-model in the
        regression tree.
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
            model = train_new_model(run, inter, NN_complexity[i][j],
                                    tmp_records[i][j])
            trained_index[i].append(model)
            # load data into array somewhere

            if i < M:
                for r in tmp_records[i][j]:
                    p = int(trained_index[i][j].predict(r.key)/stages[i+1])
                    tmp_records[i+1][p].add(r)

    # if error too high (above threshold) replace with b-tree here

    return trained_index


def train_new_model(run, inter, train_dataset, train_labels):
    model = keras.Sequential([
            layers.Dense(32, activation=tf.nn.relu, input_shape=[1]),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(1)
            ])
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    max_epochs = EPOCHS
    callbacks = [keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                               min_delta=0, patience=10,
                                               verbose=0, mode='auto',
                                               baseline=None),
                 keras.callbacks.ModelCheckpoint(
                     '{}/weights{}_{}.h5'.format(WEIGHTS_PATH, run, inter),
                     monitor='mean_absolute_error',
                     verbose=0, save_best_only=True,
                     save_weights_only=True,
                     mode='auto', period=1)]
    model.fit(train_dataset, train_labels, epochs=max_epochs,
              validation_split=1,
              verbose=0, callbacks=callbacks)
    return model


if __name__ == "__main__":
    for i in range(config.N_RUNS):
        for j in range(config.N_INTERPOLATIONS):
            index = train_hybrid([1, 2, 3], '{}/run{}inter{}'.format(
                DATA_PATH, i, j), i, j, None, 0)
            # TODO: grid search
