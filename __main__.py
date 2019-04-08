import time

import config
import util.visualiser as visualiser
import logging
import util.data_generator as datagen
import index.array_index as array
import numpy as np
import pandas as pd
import index.naive_learned_index as li
import util.access as access


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.debug("Start predictions...")

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # LI predictions -------------------------------------

    li_predictions = []
    li_pred_times = []
    li_search_times = []
    for run in range(0, config.N_RUNS):
        print("run # " + str(run + 1) + "/" + str(config.N_RUNS))
        li_predictions.append([])
        li_pred_times.append([])
        li_search_times.append([])
        if run < 0:
            continue
        for inter in range(0,config.N_INTERPOLATIONS):
            # load data for this run and interpolation
            data = datagen.load_integer_data(run, inter)

            print("inter #{}/{}".format((inter + 1), config.N_INTERPOLATIONS))

            import os.path
            if not os.path.isfile(config.MODEL_PATH +
                                  "weights{}_{}.h5".format(run, inter)):
                raise FileNotFoundError(
                    "No model trained for run{}inter{}".format(run, inter))

            dataset_path = config.DATASET_PATH+'run{}inter{}'.format(run, inter)
            cp_name = config.MODEL_PATH + 'weights{}_{}.h5'.format(run, inter)

            learned_index = li.Model([32, 32], dataset_path, cp_name,
                                     config.STEP_SIZE)
            learned_index.load_weights(cp_name)

            step = int(config.N_KEYS / config.N_SAMPLES)

            predictions = {}
            tic_pred = time.time()
            for key in range(0, config.N_KEYS, step):
                predictions[key]=learned_index.predict(key)
            toc_pred = time.time()

            inter_prediction_reads = []
            tic_search = time.time()
            for key in range(0, config.N_KEYS, step):
                reads = access.get_accesses(data, predictions[key], key)
                inter_prediction_reads.append(reads)
                data.reset_access_count()
            toc_search = time.time()

            li_predictions[run].append(inter_prediction_reads)
            li_pred_times[run].append(toc_pred-tic_pred)
            li_search_times[run].append(toc_search-tic_search)
    print("Done.")

    # ---------------- process time -------------------------------------------

    # naive_pred_times = np.average(li_pred_times, axis=0)
    naive_pred_times = np.asarray(li_pred_times).flatten(order='C')
    naive_pred_times = np.divide(naive_pred_times, config.N_SAMPLES)

    naive_search_times = np.asarray(li_search_times).flatten(order='C')
    naive_search_times = np.divide(naive_search_times, config.N_SAMPLES)

    # get microseconds
    naive_pred_times = np.multiply(naive_pred_times, 1000000)
    naive_search_times = np.multiply(naive_search_times, 1000000)

    # save time
    pd.DataFrame(naive_pred_times).to_csv(config.PREDICTIONS_PATH +
                                          "pred_times.csv", header=None)
    pd.DataFrame(naive_search_times).to_csv(config.PREDICTIONS_PATH +
                                            "search_times.csv", header=None)

    # ---------------- process reads ------------------------------------------

    # naive_efficiency = np.average(np.average(li_predictions, axis=0), axis=1)
    a = np.average(li_predictions, axis=2)
    naive_efficiency = a.flatten(order='C')

    # save reads
    pd.DataFrame(naive_efficiency).to_csv(config.PREDICTIONS_PATH +
                                          "reads.csv", header=None)

    # plot all
    visualiser.show("scatter", "", "")
    visualiser.show("hist2d", "", "")


if __name__ == "__main__":
    main()
