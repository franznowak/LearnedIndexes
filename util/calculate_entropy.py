# --------------------------------------------------------------------
# entropy.py - allows for the calculation of an approximation of how noise a
# dataset.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------
import csv
import math


def calculate_entropy(filename):
    """
    Calculates an approximation for the noisiness of a data file.

    :param filename: a csv file with keys and indexes.

    :return: a float value approximating the entropy

    """
    key_counts = {}
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            key = int(row[0])
            if key not in key_counts.keys():
                key_counts[key] = 1
            else:
                key_counts[key] = key_counts[key] + 1
    total_keys = len(key_counts.keys())
    jumps = {}
    for key, val in key_counts.items():
        if val not in jumps.keys():
            jumps[val] = 1.0
        else:
            jumps[val] = jumps[val] + 1.0

    # jumps_counts = sorted(jumps.items(), key=lambda kv: (kv[1], kv[0]))
    # # print(jumps_counts)
    # (most_common, count) = jumps_counts[len(jumps_counts)-1]
    # p = count/total_keys
    # # print("most common was {} with a ratio of {} ({}/{})".format(most_common,
    # #                                                       p, count, total_keys))
    #
    # if p == 0 or p == 1:
    #     return 0
    # else:
    #     return - p * math.log2(p) - (1-p) * math.log2(1-p)

    # H = 0
    # for key, val in jumps.items():
    #     p = val/total_keys
    #     H -= p * math.log2(p)
    # return H
    print(jumps)
    return 1-jumps[1]/total_keys


if __name__ == "__main__":
    for i in range(10):
        entropy = calculate_entropy(
            "../data/datasets/Integers_100x10x100k/run0inter{}".format(i))
        print(str(entropy))
    print()
    entropy = calculate_entropy(
        "../data/datasets/Creditcard_285k/creditcard.csv_training")
    print(str(entropy))
