import time

import config
import util.data_generator as datagen
import index.array_index as array
import numpy as np
import index.naive_learned_index as li
import util.access as access
from matplotlib import pyplot as plt


def main():
    print("loading data...")
    all_data = datagen.load_all_data()
    print("Done.")

    print("Start predictions...")



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

            tic = time.time()

            print("inter # " + str(inter + 1) + "/" + str(config.N_INTERPOLATIONS))
            inter_prediction = []
            try:
                keys, index_predictions = li.predict(run, inter)
            except FileNotFoundError as e:
                print("Skipped model {}_{}".format(run, inter))
                li_predictions[run].append(inter_prediction)
                continue
            step = int(config.N_KEYS / config.N_SAMPLES)
            for key in range(0, config.N_KEYS, step):
                data = all_data[run][inter]
                prediction = int(index_predictions[key])
                inter_prediction.append(access.get_accesses(data, prediction,
                                                      key))
                data.reset_access_count()

            toc = time.time()

            li_predictions[run].append(inter_prediction)
            li_pred_times[run].append(toc-tic)
    print("Done.")

    x = np.arange(0, config.N_INTERPOLATIONS)
    x = np.tile(x,config.N_RUNS)

    # ---plot time-----
    # naive_times = np.average(li_pred_times, axis=0)
    naive_times = np.asarray(li_pred_times).flatten(order='C')
    naive_times = np.divide(naive_times, config.N_SAMPLES)
    #plt.plot(x,naive_times)
    plt.scatter(x, naive_times)
    plt.show()

    # --------------- plot reads ----------------------------------------------
    # naive_efficiency = np.average(np.average(li_predictions, axis=0), axis=1)
    a = np.average(li_predictions, axis=2)
    naive_efficiency = a.flatten(order='C')
    fig, ax = plt.subplots()
    #ax.plot(x, naive_efficiency)
    ax.set(xlabel='entropy', ylabel='number of reads',
           title='Access times')
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
    # ------------------------------------------------------------------------

    # --------------------- plot times ---------------------------------------


if __name__ == "__main__":
    main()
