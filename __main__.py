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

# # evaluation.measure_predictions_on_synthetic_integers("naive_learned_index")
#
# data = NumKeyValData()
# dataset_name = config.REAL_WORLD_DATASET + \
#                "/creditcard.csv_training"
#
# dataset_file = config.DATASET_PATH + dataset_name
# data.load(dataset_file)
# model_path = config.MODEL_PATH + "recursive_learned_index" + "/" + \
# dataset_name + "/"
# inter_prediction_reads, prediction_time, search_time = \
#                 evaluation.evaluate_recursive_learned_index(data,
#                                                             dataset_file,
#                                                             model_path)
# print(inter_prediction_reads)

evaluation.measure_predictions_on_synthetic_integers("recursive_learned_index")
