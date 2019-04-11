# --------------------------------------------------------------------
# integer_data_generator.py - functions for generating synthetic integer data
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import config
import logging
from util.datatypes import NumKeyValData

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_integer_data():
    """
    Generates multiple integer datasets with different levels of entropy.

    :return: a list of runs containing a list of interpolations of integer
    data

    """
    logger.debug("generating all integer data in the specified range.")

    all_data = []
    for run in range(config.N_RUNS):
        print("creating data for run # " + str(run+1) + "/" + str(
            config.N_RUNS))
        run_data = []
        seed = run
        for interpolation in range(config.N_INTERPOLATIONS):
            n_keys = config.N_KEYS
            ratio = 1/config.N_INTERPOLATIONS * interpolation
            interpolation_data = NumKeyValData()
            interpolation_data.create(seed, n_keys, ratio)
            run_data.append(interpolation_data)

            file_name = "run" + str(run) + "inter" + str(interpolation)
            interpolation_data.save(config.INTEGER_DATASET_PATH + file_name)

        all_data.append(run_data)
    return all_data


def load_integer_data(run, inter):
    """
    Loads and returns the dataset for a specific run and interpolation.

    :param run: the generation from which to return data
    :param inter: the randomness level of the data

    :return: the previously generated dataset of the specified run and
    interpolation.

    """
    logger.debug("loading data for run {} inter {}.".format(run, inter))

    data = NumKeyValData()
    file_name = "run" + str(run) + "inter" + str(inter)
    data.load(config.INTEGER_DATASET_PATH + file_name)
    return data


def load_all_integer_data():
    """
    Loads and returns all the integer data for all runs and entropy levels.

    :return: a list of runs of lists of integer data

    """
    all_data = []
    for run in range(config.N_RUNS):
        logger.debug("loaded {}/{} runs.".format(run+1, config.N_RUNS))

        run_data = []
        for interpolation in range(config.N_INTERPOLATIONS):
            interpolation_data = load_integer_data(run, interpolation)
            run_data.append(interpolation_data)

        all_data.append(run_data)
    return all_data
