# --------------------------------------------------------------------
# prediction.py - class for making predictions using index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import time
import config
import logging
import util.data_generator as datagen
import numpy as np
import os
from index.array_index import ArrayIndex
from index.btree_index import BTreeSet
from index.naive_learned_index import Model
from index.recursive_learned_index import RecursiveLearnedIndex
import util.search as searcher
from custom_exceptions import NoEvaluationImplemented
from util.datatypes import NumKeyValData

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def measure_predictions_on_synthetic_integers(index_type, n_runs=config.N_RUNS):
    """
    Measures prediction times for runs and interpolations of synthetic Integer
    data and saves them to file.

    :param index_type: type of index to be used
    :param n_runs: number of runs of synthetic integer data to evaluate the
    index on

    """
    logger.debug("Start predictions...")

    n_reads = []
    prediction_times = []
    search_times = []

    if n_runs == 0:
        # No runs so abort
        return

    for run in range(0, n_runs):

        print("run #" + str(run))

        n_reads.append([])
        prediction_times.append([])
        search_times.append([])

        for inter in range(0, config.N_INTERPOLATIONS):

            print("inter #" + str(inter))

            data = datagen.load_integer_data(run, inter)

            # load model data if it is a learned index
            dataset_file = config.INTEGER_DATASET_PATH + \
                'run{}inter{}'.format(run, inter)

            if index_type == "recursive_learned_index":
                model_file = config.MODEL_PATH + index_type + "/" + \
                             config.INTEGER_DATASET + \
                             '/run{}inter{}/'.format(run, inter)
            else:
                model_file = config.MODEL_PATH + index_type + "/" + \
                            config.INTEGER_DATASET + \
                            '/weights{}_{}.h5'.format(run, inter)

            inter_prediction_reads, prediction_time, search_time = \
                get_prediction_times(index_type, data, dataset_file, model_file)

            n_reads[run].append(inter_prediction_reads)
            prediction_times[run].append(prediction_time)
            search_times[run].append(search_time)

    print("Done.")

    # process time and reads

    pred_times = np.asarray(prediction_times).transpose()
    pred_times = np.multiply(pred_times, 1000000)  # microseconds

    search_times = np.asarray(search_times).transpose()
    search_times = np.multiply(search_times, 1000000)  # microseconds

    total_times = np.add(pred_times, search_times)

    n_reads = np.asarray(n_reads).transpose()

    # Set where predictions are stored
    prediction_path = config.PREDICTIONS_PATH + index_type + "/" + \
        config.INTEGER_DATASET + "/"
    if not os.path.isdir(prediction_path):
        os.makedirs(prediction_path)

    # save time
    save_predictions(pred_times, prediction_path, "pred_times")
    save_predictions(search_times, prediction_path, "search_times")
    save_predictions(total_times, prediction_path, "total_times")
    save_predictions(n_reads, prediction_path, "reads")


def measure_predictions_on_single_dataset(index_type, dataset, file):
    dataset_path = config.DATASET_PATH + "/" + dataset + "/" + file
    data = NumKeyValData()
    data.load(dataset_path)

    if index_type == "recursive_learned_index":
        rwd_model_path = config.MODEL_PATH + index_type + "/" + dataset + "/"
    else:
        rwd_model_path = config.MODEL_PATH + index_type + "/" + \
                         config.REAL_WORLD_DATASET + "/weights.h5"

    prediction_reads, prediction_time, search_time = \
        get_prediction_times(index_type, data, dataset_path, rwd_model_path)

    prediction_time = prediction_time * 1000000  # microseconds
    search_time = search_time * 1000000  # microseconds

    total_time = prediction_time + search_time

    # Set where predictions are stored
    prediction_path = config.PREDICTIONS_PATH + index_type + "/" + \
        dataset + "/"
    if not os.path.isdir(prediction_path):
        os.makedirs(prediction_path)

    # Save predictions
    save_predictions([[prediction_time]], prediction_path, "pred_times")
    save_predictions([[search_time]], prediction_path, "search_times")
    save_predictions([[total_time]], prediction_path, "total_times")
    save_predictions([[prediction_reads]], prediction_path, "reads")


def get_prediction_times(index_type, data, dataset_file="", model_file=""):
    """
    Returns the prediction times for a single dataset evaluated using a
    specified index_type.

    :param index_type: string - type of index to use
    :param data: NumKeyValData to count accesses
    :param dataset_file: filename of file containing the data
    :param model_file: filename or path of file(s) containing the model
    weights for the learned index.

    :return:  prediction_reads, prediction_time, search_time

    """
    if index_type == "array_index":
        return evaluate_array_index(data)
    elif index_type == "binary_search":
        return evaluate_binary_search(data)
    elif index_type == "btree_index":
        return evaluate_btree_index(data)
    elif index_type == "naive_learned_index":
        return evaluate_naive_learned_index(data, dataset_file, model_file)
    elif index_type == "recursive_learned_index":
        return evaluate_recursive_learned_index(data, dataset_file, model_file)
    else:
        raise NoEvaluationImplemented(index_type)


def evaluate_array_index(data):
    """
    Evaluates the array index according to config settings.

    :param data: NumKeyValData to count accesses
    :return: prediction_reads, prediction_time, search_time

    """
    step = int(data.size / config.N_SAMPLES)

    predictions = {}

    tic_pred = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        predictions[key] = ArrayIndex.predict(data, key)
    toc_pred = time.time()

    inter_prediction_reads = []

    tic_search = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        reads = get_search_access_count(data, predictions[key], key)
        inter_prediction_reads.append(reads)
        data.reset_access_count()
    toc_search = time.time()

    prediction_time = (toc_pred - tic_pred) / config.N_SAMPLES
    search_time = (toc_search - tic_search) / config.N_SAMPLES
    inter_prediction_reads = np.average(inter_prediction_reads)

    return inter_prediction_reads, prediction_time, search_time


def evaluate_binary_search(data):
    """
    Evaluates the binary search according to config settings.

    :param data: NumKeyValData to count accesses
    :return: prediction_reads, prediction_time, search_time

    """
    step = int(data.size / config.N_SAMPLES)
    inter_prediction_reads = []

    tic_search = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        searcher.binary_search(data, key)
        reads = data.get_access_count()
        inter_prediction_reads.append(reads)
        data.reset_access_count()
    toc_search = time.time()

    prediction_time = 0
    search_time = (toc_search - tic_search) / config.N_SAMPLES
    inter_prediction_reads = np.average(inter_prediction_reads)

    return inter_prediction_reads, prediction_time, search_time


def evaluate_btree_index(data):
    """
    Evaluates the btree index according to config settings.

    :param data: NumKeyValData to count accesses
    :return: prediction_reads, prediction_time, search_time

    """
    btree = BTreeSet(4)
    for i in range(data.size):
        btree.add(data.data_array[i])

    step = int(data.size / config.N_SAMPLES)

    inter_prediction_reads = []

    tic_search = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        reads = btree.count_reads_to(key)
        inter_prediction_reads.append(reads)
    toc_search = time.time()

    prediction_time = (toc_search - tic_search) / config.N_SAMPLES
    search_time = 0
    inter_prediction_reads = np.average(inter_prediction_reads)

    return inter_prediction_reads, prediction_time, search_time


def evaluate_naive_learned_index(data, dataset_file, model_file):
    """
    Evaluates the naive learned index according to config settings.

    :param data: NumKeyValData to count accesses
    :param dataset_file: file where data is stored
    :param model_file: file where weights are stored
    :return: prediction_reads, prediction_time, search_time

    """
    training_data = Model.load_training_data(dataset_file)
    naive_index = Model(config.NAIVE_COMPLEXITY, training_data, model_file)
    naive_index.load_weights(model_file)

    step = int(data.size / config.N_SAMPLES)

    predictions = {}

    tic_pred = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        predictions[key] = naive_index.predict(key)
    toc_pred = time.time()

    inter_prediction_reads = []

    tic_search = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        reads = get_search_access_count(data, predictions[key], key)
        inter_prediction_reads.append(reads)
        data.reset_access_count()
    toc_search = time.time()

    prediction_time = (toc_pred - tic_pred) / config.N_SAMPLES
    search_time = (toc_search - tic_search) / config.N_SAMPLES
    inter_prediction_reads = np.average(inter_prediction_reads)

    return inter_prediction_reads, prediction_time, search_time


def evaluate_recursive_learned_index(data, dataset_file, model_path):
    """
    Evaluates the recursive learned index according to config settings.

    :param data: NumKeyValData to count accesses
    :param dataset_file: file where data is stored
    :param model_path: path where weight files and stat files are stored.
    :return: prediction_reads, prediction_time, search_time

    """
    training_data = Model.load_training_data(dataset_file)
    recursive_index = RecursiveLearnedIndex(config.RECURSIVE_SHAPE,
                                            config.RECURSIVE_COMPLEXITY)
    recursive_index.load_models(training_data, model_path)

    step = int(data.size / config.N_SAMPLES)

    predictions = {}

    tic_pred = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        predictions[key] = recursive_index.predict(key)
    toc_pred = time.time()

    inter_prediction_reads = []

    tic_search = time.time()
    for j in range(0, data.size, step):
        key = data.data_array[j]
        reads = get_search_access_count(data, predictions[key], key)
        inter_prediction_reads.append(reads)
        data.reset_access_count()
    toc_search = time.time()

    prediction_time = (toc_pred - tic_pred) / config.N_SAMPLES
    search_time = (toc_search - tic_search) / config.N_SAMPLES
    inter_prediction_reads = np.average(inter_prediction_reads)

    return inter_prediction_reads, prediction_time, search_time


def save_predictions(data, path, filename):
    """
    Saves two copies of the prediction data to file, one with timestamp,
    one as the newest available data.

    :param data: data to be saved.
    :param path: path to directory in which to store the data
    :param filename: identifier for file, signifying kind of data being stored

    """
    # filename with timestamp
    fn = "{}_{}.csv".format(int(time.time()), filename)
    write_predictions_to_file(data,  path, fn)
    # filename for latest available data
    fn2 = "new_{}.csv".format(filename)
    write_predictions_to_file(data, path, fn2)


def write_predictions_to_file(data, path, filename):
    """
    Helper function that writes the data to file.

    :param data: data to be saved
    :param path: path to directory in which to store the data
    :param filename: filename of the file in which to store the data

    """
    with open(path + filename, mode='w') as file:
        for i in range(len(data)):
            for j in range(len(data[i])):
                file.write("{},{}\n".format(i, data[i][j]))


def get_search_access_count(input_data, prediction, key):
    """
    Searches for the key and returns the data accesses that were necessary to
    find it.

    :param input_data: the data of type NumKeyValData
    :param prediction: the estimated position of the key in the data
    :param key: the key to be found

    :return: number of accesses used in search

    """
    searcher.search(input_data, prediction, key)
    return input_data.get_access_count()
