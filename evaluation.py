# --------------------------------------------------------------------
# prediction.py - class for making predictions using index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import time
import config
import util.visualiser as visualiser
import logging
import util.integer_data_generator as datagen
import numpy as np
import os
from index.array_index import ArrayIndex
from index.btree_index import BTreeSet
from index.naive_learned_index import Model
from index.recursive_learned_index import RecursiveLearnedIndex
import util.search as searcher
from custom_exceptions import NoEvaluationImplemented

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def measure_predictions_on_synthetic_integers(index_type):
    """
    Measures prediction times for runs and interpolations of synthetic Integer
    data and saves them to file.

    """
    logger.debug("Start array predictions...")

    n_reads = []
    prediction_times = []
    search_times = []

    for run in range(0, config.N_RUNS):

        print("run # " + str(run + 1) + "/" + str(config.N_RUNS))

        n_reads.append([])
        prediction_times.append([])
        search_times.append([])

        for inter in range(0, config.N_INTERPOLATIONS):

            print("inter #{}/{}".format((inter + 1), config.N_INTERPOLATIONS))

            data = datagen.load_integer_data(run, inter)

            dataset_file = model_file = ""
            # load model data if it is a naive learned index
            if index_type == "naive_learned_index":
                dataset_file = config.INTEGER_DATASET_PATH \
                    + 'run{}inter{}'.format(run, inter)
                if not os.path.isfile(model_file):
                    raise FileNotFoundError(
                        "No data found at {}".format(dataset_file))
                model_file = \
                    config.MODEL_PATH + index_type + config.INTEGER_DATASET +\
                    '/weights{}_{}.h5'.format(run, inter)
                if not os.path.isfile(model_file):
                    raise FileNotFoundError(
                        "No model trained for {} run{}inter{}".format(
                            index_type, run, inter))

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

    naive_efficiency = np.average(n_reads, axis=2).transpose()

    prediction_path = config.PREDICTIONS_PATH + index_type \
        + config.INTEGER_DATASET + "/"
    if not os.path.isdir(prediction_path):
        os.mkdir(prediction_path)

    # save time
    save_predictions(pred_times, prediction_path, "pred_times")
    save_predictions(search_times, prediction_path, "search_times")
    save_predictions(total_times, prediction_path, "total_times")
    save_predictions(naive_efficiency, prediction_path, "reads")

    # plot all
    visualiser.show(prediction_path, "scatter")
    visualiser.show(prediction_path, "hist2d")


def get_prediction_times(index_type, data, dataset_file="", model_file=""):

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
    step = int(config.N_KEYS / config.N_SAMPLES)

    predictions = {}

    tic_pred = time.time()
    for key in range(0, config.N_KEYS, step):
        predictions[key] = ArrayIndex.predict(data, key)
    toc_pred = time.time()

    inter_prediction_reads = []

    tic_search = time.time()
    for key in range(0, config.N_KEYS, step):
        reads = get_search_access_count(data, predictions[key], key)
        inter_prediction_reads.append(reads)
        data.reset_access_count()
    toc_search = time.time()

    prediction_time = (toc_pred - tic_pred) / config.N_SAMPLES
    search_time = (toc_search - tic_search) / config.N_SAMPLES

    return inter_prediction_reads, prediction_time, search_time


def evaluate_binary_search(data):
    step = int(config.N_KEYS / config.N_SAMPLES)
    inter_prediction_reads = []

    tic_search = time.time()
    for key in range(0, config.N_KEYS, step):
        searcher.binary_search(data, key)
        reads = data.get_access_count()
        inter_prediction_reads.append(reads)
        data.reset_access_count()
    toc_search = time.time()

    prediction_time = 0
    search_time = (toc_search - tic_search) / config.N_SAMPLES

    return inter_prediction_reads, prediction_time, search_time


def evaluate_btree_index(data):
    btree = BTreeSet(64)
    for i in range(0, len(data.data_array), 2):
        btree.add(data.data_array[i])

    step = int(config.N_KEYS / config.N_SAMPLES)

    inter_prediction_reads = []

    tic_search = time.time()
    for key in range(0, config.N_KEYS, step):
        reads = btree.count_reads_to(key)
        inter_prediction_reads.append(reads)
    toc_search = time.time()

    prediction_time = 0
    search_time = (toc_search - tic_search) / config.N_SAMPLES

    return inter_prediction_reads, prediction_time, search_time


def evaluate_naive_learned_index(data, dataset_file, model_file):
    training_data = Model.load_training_data(dataset_file)
    index = Model([32, 32], training_data, model_file)
    index.load_weights(model_file)

    step = int(config.N_KEYS / config.N_SAMPLES)

    predictions = {}

    tic_pred = time.time()
    for key in range(0, config.N_KEYS, step):
        predictions[key] = index.predict(key)
    toc_pred = time.time()

    inter_prediction_reads = []

    tic_search = time.time()
    for key in range(0, config.N_KEYS, step):
        reads = get_search_access_count(data, predictions[key], key)
        inter_prediction_reads.append(reads)
        data.reset_access_count()
    toc_search = time.time()

    prediction_time = (toc_pred - tic_pred) / config.N_SAMPLES
    search_time = (toc_search - tic_search) / config.N_SAMPLES

    return inter_prediction_reads, prediction_time, search_time


def evaluate_recursive_learned_index(data, dataset_file, model_path):
    raise NoEvaluationImplemented
    # recursive_index = RecursiveLearnedIndex(config.RECURSIVE_SHAPE,
    #                                         config.RECURSIVE_COMPLEXITY)
    # training_data = Model.load_training_data(dataset_file)


def save_predictions(data, path, filename):
    """
    Saves two copies of the prediction data to file, one with timestamp,
    one as the newest available data.

    :param data: data to be saved.
    :param path: path to directory in which to store the data
    :param filename: identifier for file, signifying kind of data being stored

    """
    # filename with timestamp
    fn = "{}{}_{}.csv".format(path, int(time.time()), filename)
    write_predictions_to_file(data,  path, fn)
    # filename for latest available data
    fn2 = "{}new_{}.csv".format(path, filename)
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
