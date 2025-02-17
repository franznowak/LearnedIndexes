# --------------------------------------------------------------------
# recursive_train_model - script to train the recursive learned index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import os
import config
from index.recursive_learned_index import RecursiveLearnedIndex
from index.naive_learned_index import Model

EPOCHS = 1000
DATA_SET = config.REAL_WORLD_DATASET
DATA_PATH = "../data/datasets/" + DATA_SET + "/"
GRAPH_PATH = "../data/graphs/recursive_learned_index/" + DATA_SET + "/"
FILENAME = "creditcard.csv_training"

print("Training on {}".format(FILENAME))

WEIGHTS_PATH = "../data/indexes/recursive_learned_index/" + DATA_SET + "/" + \
               FILENAME + "/"

if not os.path.isdir(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)

learned_index = RecursiveLearnedIndex(config.RECURSIVE_SHAPE,
                                      config.RECURSIVE_COMPLEXITY)
training_data = Model.load_training_data(DATA_PATH + FILENAME)
learned_index.train(training_data, WEIGHTS_PATH, GRAPH_PATH, EPOCHS)
