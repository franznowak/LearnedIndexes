# --------------------------------------------------------------------
# entropy.py - allows for the calculation of an approximation of how noise a
# data set.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------
import csv
import math


def markov_entropy_order_zero(filename, weighted=True):
    """
    Calculates a simple approximation for the noisiness of a data file based
    on the bucket frequencies (the key-frequency frequencies).

    :param filename: a csv file with integer keys (and any records) sorted by
    key in ascending order.
    :param weighted: if True, weights the entropy by the bucket size.

    :return: entropy - a float value approximating the entropy of the data set

    """
    key_counts = _get_key_counts(filename)
    bucket_probabilities = _get_bucket_probabilities(key_counts)

    entropy = 0
    # H = -Sum_i(p(i) * log_2 p(i))
    for i, p_i in bucket_probabilities.items():
        if weighted:
            entropy -= i * p_i * math.log2(p_i)
        else:
            entropy -= p_i * math.log2(p_i)

    return entropy


def markov_entropy_order_one(filename, weighted=True):
    """
    Calculates a first order Markov approximation for the noisiness of a data
    file based on the bucket frequencies (the key-frequency frequencies).

    :param filename: a csv file with integer keys (and any records) sorted by
    key in ascending order
    :param weighted: if True, weights the entropy by the bucket size.

    :return: entropy - a float value approximating the entropy of the data set

    """
    key_counts = _get_key_counts(filename)
    bucket_probabilities = _get_bucket_probabilities(key_counts)
    transition_probabilities = _get_bucket_transition_probabilities(key_counts)

    entropy = 0
    # H = -Sum_i(p(i) * Sum_j(p_i(j) * log_2(p_i(j))))
    for i, p_i in bucket_probabilities.items():
        for j, p_ij in transition_probabilities[i].items():
            if weighted:
                entropy -= i * p_i * p_ij * math.log2(p_ij)
            else:
                entropy -= p_i * p_ij * math.log2(p_ij)

    return entropy


def markov_step_entropy_order_zero(filename):
    """
    Calculates a simple approximation for the noisiness of a data file using
    the steps between keys.

    :param filename: a csv file with integer keys (and any records) sorted by
    key in ascending order.

    :return: a float value approximating the entropy of the data set

    """
    key_list = _get_step_list(filename)
    key_probabilities = _get_key_probabilities(key_list)

    entropy = 0
    # H = -Sum_i(p(i) * log_2 p(i))
    for i, p_i in key_probabilities.items():
        entropy -= p_i * math.log2(p_i)

    return entropy


def markov_step_entropy_order_one(filename):
    """
    Calculates a first order Markov approximation for the noisiness of a data
    file using the steps between keys.

    :param filename:
    :return: entropy - a float value approximating the entropy of the data set

    """
    key_list = _get_step_list(filename)
    key_probabilities = _get_key_probabilities(key_list)
    transition_probabilities = _get_bucket_transition_probabilities(key_list)

    entropy = 0
    # H = -Sum_i(p(i) * Sum_j(p_i(j) * log_2(p_i(j))))
    for i, p_i in key_probabilities.items():
        try:
            for j, p_ij in transition_probabilities[i].items():
                entropy -= p_i * p_ij * math.log2(p_ij)
        except Exception:
            pass

    return entropy


def _get_key_counts(filename):
    """
    Counts the number of occurrences of each key in a data set.

    :param filename: a csv file with integer keys (and any records) sorted by
    key in ascending order.

    :return: key_counts - dict with keys as keys and their occurrence count in
    the data set as values

    """
    # Count occurrence frequencies of all keys
    key_counts = {}
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            key = int(row[0])
            if key not in key_counts.keys():
                key_counts[key] = 1
            else:
                key_counts[key] = key_counts[key] + 1

    # Add missing keys to the dict as frequency 0
    highest_key = max(key_counts.keys())
    for i in range(highest_key + 1):
        if i not in key_counts.keys():
            key_counts[i] = 0

    return key_counts


def _get_bucket_probabilities(key_counts):
    """
    Calculates the probabilities p_i of each bucket.

    :param key_counts: counts of occurrences of each key
    :return: bucket_probabilities - dict of bucket probabilities

    """
    # Count jump frequencies
    bucket_counts = {}
    for key, val in key_counts.items():
        if val not in bucket_counts.keys():
            bucket_counts[val] = 1.0
        else:
            bucket_counts[val] = bucket_counts[val] + 1.0

    # Divide by total keys to get probabilities
    bucket_probabilities = {}
    n_keys = len(key_counts.keys())
    for i, n_i in bucket_counts.items():
        bucket_probabilities[i] = n_i/n_keys

    return bucket_probabilities


def _get_bucket_transition_probabilities(key_counts):
    """
    Calculates the transition probabilities p_i(j) of going from bucket A to
    bucket B.

    :param key_counts:
    :return: transition_probabilities - dict of dicts with transition probs

    """
    # Count frequencies of jump/bucket A being followed by jump/bucket B
    transition_counts = {}
    for i in range(len(key_counts)-1):
        bucket = key_counts[i]
        nxt_bucket = key_counts[i + 1]
        if bucket not in transition_counts:
            transition_counts[bucket] = {}
        if nxt_bucket not in transition_counts[bucket]:
            transition_counts[bucket][nxt_bucket] = 1
        else:
            transition_counts[bucket][nxt_bucket] += 1

    # Count transitions
    n_transitions = 0
    for i in transition_counts.keys():
        for j, n_j in transition_counts[i].items():
            n_transitions += transition_counts[i][j]

    # Divide by total transitions to get probabilities
    transition_probabilities = {}
    for i in transition_counts.keys():
        transition_probabilities[i] = {}
        for j, n_j in transition_counts[i].items():
            transition_probabilities[i][j] = n_j/n_transitions

    return transition_probabilities

def _get_step_list(filename):
    """
    Gets a sequential list of keys from a data set.

    :param filename: a csv file with integer keys (and any records) sorted by
    key in ascending order.

    :return: key_lsit - list with keys

    """
    key_list = []
    step_list = []
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            key = int(row[0])
            key_list.append(key)
            step_list.append(key - key_list[len(key_list)-2])
    return step_list


def _get_key_probabilities(key_list):
    """
    Calculates the probabilities p_i of each key.

    :param key_list: a list of keys
    :return: key_probabilities - dict of key probabilities

    """
    # Count occurrence frequencies of all keys
    key_counts = {}
    for key in key_list:
        if key not in key_counts.keys():
            key_counts[key] = 1
        else:
            key_counts[key] = key_counts[key] + 1

    # Divide by total number of keys to get probabilities
    key_probabilities = {}
    for i, n_i in key_counts.items():
        key_probabilities[i] = n_i/len(key_list)

    return key_probabilities


if __name__ == "__main__":
    SYNTH_PATH = "../data/datasets/Integers_100x10x100k/run0inter"
    RWD_PATH = "../data/datasets/Creditcard_285k/creditcard.csv_training"

    print("Synth Entropy:")
    for interpolation in range(10):
        print(markov_entropy_order_zero(
            "{}{}".format(SYNTH_PATH, interpolation)))
    print()
    print("RWD Entropy 0: {}".format(markov_entropy_order_zero(
        RWD_PATH)))

    print("Synth Entropy:")
    for interpolation in range(10):
        print(markov_entropy_order_one("{}{}".format(SYNTH_PATH,
                                                      interpolation)))
    print()
    print("RWD Entropy 1: {}".format(markov_entropy_order_one(RWD_PATH)))
