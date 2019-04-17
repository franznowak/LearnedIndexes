# --------------------------------------------------------------------
# __main__.py - script for evaluation all indexes for time and file accesses.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------
import os

import config
import evaluation
from util import visualiser
from util.datatypes import NumKeyValData

"""
Systematically test all indexes and save the prediction times and accesses
for evaluation.

"""

index_types = ["array_index", "binary_search", "btree_index",
               "naive_learned_index", "recursive_learned_index"]

RWD_DATASET_NAME = config.REAL_WORLD_DATASET + "/creditcard.csv_training"
RWD_DATASET_FILE = config.DATASET_PATH + RWD_DATASET_NAME

# Evaluation on synthetic integer data
for index_type in index_types:
    print("{} evaluation on synthetic Integer data".format(index_type))
    try:
        if index_type == "recursive_learned_index":
            evaluation.measure_predictions_on_synthetic_integers(index_type,
                                                                 2)
        else:
            evaluation.measure_predictions_on_synthetic_integers(index_type,
                                                                 100)

        # Set where graphs are stored
        graph_path = config.GRAPH_PATH + index_type + "/" + \
            config.INTEGER_DATASET + "/"
        if not os.path.isdir(graph_path):
            os.makedirs(graph_path)

        prediction_path = config.PREDICTIONS_PATH + index_type + "/" + \
            config.INTEGER_DATASET + "/"

        # plot all
        visualiser.create_graphs(prediction_path, graph_path, kind="line")

    except Exception as e:
        print("Exception occurred during evaluation of {}: {}".format(
            index_type, e.args))


# Evaluation on real world dataset


for index_type in index_types:
    try:
        print("{} evaluation on real world data".format(index_type))

    except Exception as e:
        print("Exception occurred during evaluation of {}: {}".format(
            index_type, e.args))
