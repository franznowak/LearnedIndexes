import time

import config
import logging
import util.data_generator as datagen
import index.array_index as array
import numpy as np
import matplotlib.cm as cm
import index.naive_learned_index as li
import util.access as access
from matplotlib import pyplot as plt


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.debug("Start predictions...")

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # LI predictions -------------------------------------

    li_predictions = []
    li_pred_times = []
    for run in range(0, config.N_RUNS):
        print("run # " + str(run + 1) + "/" + str(config.N_RUNS))
        li_predictions.append([])
        li_pred_times.append([])
        if run < 0:
            continue
        for inter in range(0,config.N_INTERPOLATIONS):
            # load data for this run and interpolation
            data = datagen.load_integer_data(run, inter)


            print("inter # " + str(inter + 1) + "/" + str(config.N_INTERPOLATIONS))
            inter_prediction = []

            tic_pred = time.time()
            try:
                keys, index_predictions = li.predict(run, inter)
            except FileNotFoundError as e:
                print("Skipped model {}_{}".format(run, inter))
                li_predictions[run].append(inter_prediction)
                continue
            toc_pred = time.time()

            step = int(config.N_KEYS / config.N_SAMPLES)

            tic_search = time.time()

            for key in range(0, config.N_KEYS, step):

                prediction = int(index_predictions[key])
                inter_prediction.append(access.get_accesses(data, prediction,
                                                      key))
                data.reset_access_count()

            toc_search = time.time()

            li_predictions[run].append(inter_prediction)
            li_pred_times[run].append(toc_search-tic_search)
    print("Done.")

    x = np.arange(0, config.N_INTERPOLATIONS)
    x = np.tile(x,config.N_RUNS)

    # ---plot time-----
    # naive_times = np.average(li_pred_times, axis=0)
    naive_times = np.asarray(li_pred_times).flatten(order='C')
    naive_times = np.divide(naive_times, config.N_SAMPLES) # N_KEYS for tic_pred
    # get microseconds
    naive_times = np.multiply(naive_times, 1000000)

    fig, ax = plt.subplots()
    ax.set(xlabel='entropy', ylabel='time in microseconds',
           title='Access time in microseconds')
    ax.grid()

    plt.scatter(x, naive_times)
    plt.show()
    plt.hist2d(x, naive_times, bins=50, cmap=cm.jet)
    plt.show()

    # --------------- plot reads ----------------------------------------------
    # naive_efficiency = np.average(np.average(li_predictions, axis=0), axis=1)
    a = np.average(li_predictions, axis=2)
    naive_efficiency = a.flatten(order='C')
    fig, ax = plt.subplots()
    ax.set(xlabel='entropy', ylabel='number of reads',
           title='Average reads required for search')
    ax.grid()

    array_efficiency = [0,145,200,230,250,260,280,265,270,280]
    btree_efficiency = [70 for _ in range(10)]

    A = array_efficiency
    B = btree_efficiency

    # Plotting functionality starts here
    #plt.plot(A, 'b')
    #plt.plot(B, 'C1')
    #plt.plot(naive_efficiency, 'g')

    plt.scatter(x, naive_efficiency)
    plt.show()
    plt.hist2d(x, naive_efficiency, bins=50, cmap=cm.jet)
    plt.show()

    # ------------------------------------------------------------------------

    # --------------------- plot times ---------------------------------------


if __name__ == "__main__":
    main()
