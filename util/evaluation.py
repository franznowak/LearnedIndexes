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
import index.naive_learned_index as li
import util.search as searcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def measure_synthetic_integer_predictions(index_model):
    """
    Measures prediction times for runs and interpolations of synthetic Integer
    data and saves them to file.

    :param index_model: index to be used for predictions

    """
    logger.debug("Start predictions...")

    n_reads = []
    prediction_times = []
    search_times = []

    for run in range(0, config.N_RUNS):

        print("run # " + str(run + 1) + "/" + str(config.N_RUNS))

        n_reads.append([])
        prediction_times.append([])
        search_times.append([])

        for inter in range(0, config.N_INTERPOLATIONS):

            # load data for this run and interpolation
            data = datagen.load_integer_data(run, inter)

            print("inter #{}/{}".format((inter + 1), config.N_INTERPOLATIONS))

            import os.path
            if not os.path.isfile(config.MODEL_PATH +
                                  "weights{}_{}.h5".format(run, inter)):
                raise FileNotFoundError(
                    "No model trained for run{}inter{}".format(run, inter))

            dataset_path = config.INTEGER_DATASET_PATH \
                + 'run{}inter{}'.format(run, inter)
            cp_name = config.MODEL_PATH + 'weights{}_{}.h5'.format(run, inter)

            training_data = li.Model.load_training_data(dataset_path)
            learned_index = li.Model([32, 32], training_data, cp_name)
            learned_index.load_weights(cp_name)

            step = int(config.N_KEYS / config.N_SAMPLES)

            predictions = {}

            tic_pred = time.time()
            for key in range(0, config.N_KEYS, step):
                predictions[key] = learned_index.predict(key)
            toc_pred = time.time()

            inter_prediction_reads = []

            tic_search = time.time()
            for key in range(0, config.N_KEYS, step):
                reads = get_search_access_count(data, predictions[key], key)
                inter_prediction_reads.append(reads)
                data.reset_access_count()
            toc_search = time.time()

            n_reads[run].append(inter_prediction_reads)
            prediction_times[run].append(toc_pred - tic_pred)
            search_times[run].append(toc_search - tic_search)

    print("Done.")

    # process time and reads

    pred_times = np.asarray(prediction_times).transpose()
    pred_times = np.divide(pred_times, config.N_SAMPLES)
    pred_times = np.multiply(pred_times, 1000000)  # microseconds

    search_times = np.asarray(search_times).transpose()
    search_times = np.divide(search_times, config.N_SAMPLES)
    search_times = np.multiply(search_times, 1000000)  # microseconds

    total_times = np.add(pred_times, search_times)  # TODO: general

    naive_efficiency = np.average(n_reads, axis=2).transpose()

    # save time (TODO: general)
    save_predictions(pred_times, "pred_times")
    save_predictions(search_times, "search_times")
    save_predictions(total_times, "total_times")
    save_predictions(naive_efficiency, "reads")

    # plot all (TODO: general)
    visualiser.show("scatter", config.PREDICTIONS_PATH)
    visualiser.show("hist2d", config.PREDICTIONS_PATH)


def predict():
    pass


def save_predictions(data, file):
    """
    Saves two copies of the prediction data to file, one with timestamp,
    one as the newest available data.

    :param data: data to be saved.
    :param file: identifier for the file, signifying kind of data being stored

    """
    # filename with timestamp
    fn = "{}{}_{}.csv".format(config.PREDICTIONS_PATH, int(time.time()), file)
    write_predictions_to_file(data, fn)
    # filename for latest available data
    fn2 = "{}new_{}.csv".format(config.PREDICTIONS_PATH, file)
    write_predictions_to_file(data, fn2)


def write_predictions_to_file(data, filename):
    """
    Helper function that writes the data to file.

    :param data: data to be saved
    :param filename: filename of the file in which to store the data

    """
    with open(filename, mode='w') as file:
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
