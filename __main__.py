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

    # for b in all_data[0]:
    #     b.print_truncated()

    print("Start predictions...")
    # array_predictions = []
    # for run in range(config.N_RUNS):
    #     print("run # " + str(run+1) + "/" + str(config.N_RUNS))
    #     array_predictions.append([])
    #     for inter in range(config.N_INTERPOLATIONS):
    #         #array_predictions[run].append([])
    #         prediction = []
    #         step = int(config.N_KEYS / config.N_SAMPLES)
    #         for i in range(0, config.N_KEYS, step):
    #             prediction.append(array.array_predict(all_data[run][inter], i))
    #             all_data[run][inter].reset_access_count()
    #         array_predictions[run].append(prediction)
    #
    # print(array_predictions)
    #
    # x = np.arange(0, config.N_KEYS, int(config.N_KEYS / config.N_SAMPLES))
    # y = np.average(array_predictions, axis=0)[6]
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.set(xlabel='key', ylabel='number of reads',
    #        title='Access time array predict data 6')
    # ax.grid()
    # #plt.show()

    # LI predictions -------------------------------------

    li_predictions = []
    for run in range(config.N_RUNS):
        print("run # " + str(run + 1) + "/" + str(config.N_RUNS))
        li_predictions.append([])
        for inter in range(0,config.N_INTERPOLATIONS):
            print("inter # " + str(inter + 1) + "/" + str(config.N_INTERPOLATIONS))
            inter_prediction = []
            keys, index_predictions = li.predict(run, inter)
            print(index_predictions[99999])
            step = int(config.N_KEYS / config.N_SAMPLES)
            for key in range(0, config.N_KEYS, step):
                data = all_data[run][inter]
                prediction = int(index_predictions[key])
                inter_prediction.append(access.get_accesses(data, prediction,
                                                      key))
                data.reset_access_count()
            li_predictions[run].append(inter_prediction)
    print("Done.")

    x = np.arange(0, config.N_INTERPOLATIONS)
    y = np.average(li_predictions[0], axis=1)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='key', ylabel='number of reads',
           title='Access times')
    ax.grid()
    #plt.show()


    array_efficiency = [0,145,200,230,250,260,280,265,270,280]
    btree_efficiency = [70 for _ in range(10)]

    A = array_efficiency
    B = btree_efficiency

    # Plotting functionality starts here
    plt.plot(A, 'b')
    plt.plot(B, 'C1')
    plt.plot(y, 'g')
    plt.show()


if __name__ == "__main__":
    main()
