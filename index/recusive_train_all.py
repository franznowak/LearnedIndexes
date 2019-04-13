# --------------------------------------------------------------------
# recursive_train_all - script to train the recursive learned index on all
# runs and interpolations of the synthetic integer data
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import os
import config
from index.recursive_learned_index import RecursiveLearnedIndex
from index.naive_learned_index import Model

EPOCHS = 2000
N_RUNS = 1
N_INTERPOLATIONS = 1
DATA_PATH = "../data/datasets/" + config.INTEGER_DATASET +"/"
GRAPH_PATH = "../data/graphs/recursive_learned_index/" + \
             config.INTEGER_DATASET + "/"

for run in range(N_RUNS):
    for inter in range(N_INTERPOLATIONS):
        print("Training run {} interpolation {}".format(run, inter))
        FILENAME = 'run{}inter{}'.format(run, inter)
        WEIGHTS_PATH = "../data/indexes/recursive_learned_index/" + \
                       config.INTEGER_DATASET + "/" + FILENAME + "/"
        if os.path.isdir(WEIGHTS_PATH):
            continue

        learned_index = RecursiveLearnedIndex(config.RECURSIVE_SHAPE,
                                              config.RECURSIVE_COMPLEXITY)
        training_data = Model.load_training_data(DATA_PATH + FILENAME)
        learned_index.train(training_data, WEIGHTS_PATH, GRAPH_PATH, EPOCHS)
