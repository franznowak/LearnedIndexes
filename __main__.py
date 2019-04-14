# --------------------------------------------------------------------
# __main__.py - script for evaluation all indexes for time and file accesses.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------
import config
import evaluation
from util.datatypes import NumKeyValData

"""
Systematically test all indexes and save the prediction times and accesses
for evaluation.

"""

index_types = ["array_index", "binary_search", "btree_index",
               "naive_learned_index", "recursive_learned_index"]

# Evaluation on synthetic integer data
for index_type in index_types:
    try:
        if index_type == "recursive_learned_index":
            evaluation.measure_predictions_on_synthetic_integers(index_type, 1)
        else:
            evaluation.measure_predictions_on_synthetic_integers(index_type,
                                                                 100)
    except Exception as e:
        print("Exception occurred during evaluation of {}: {}".format(
            index_type, e.args))

# # Evaluation on real world dataset
# data = NumKeyValData()
# dataset_name = config.REAL_WORLD_DATASET + \
#                "/creditcard.csv_training"
#
# dataset_file = config.DATASET_PATH + dataset_name
# data.load(dataset_file)
# model_path = config.MODEL_PATH + "recursive_learned_index" + "/" + \
#     dataset_name + "/"
#
#
# # array index
# # binary search
# # btree index
# # naive learned index
# # recursive learned index
# inter_prediction_reads, prediction_time, search_time = \
#                 evaluation.evaluate_recursive_learned_index(data,
#                                                             dataset_file,
#                                                             model_path)
#
# # save(inter_prediction_reads)
# # save(prediction_time)
# # save(search_time)
# # save(total_time)
