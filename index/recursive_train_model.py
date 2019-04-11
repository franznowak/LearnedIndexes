# --------------------------------------------------------------------
# recursive_train_model - script to train the recursive learned index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

from index.recursive_learned_index import RecursiveLearnedIndex
from index.naive_learned_index import Model

EPOCHS = 2000
DATA_SET = "Integers_100x10x100k/"
DATA_PATH = "../data/datasets/" + DATA_SET
GRAPH_PATH = "../data/graphs/recursive_learned_index/" + DATA_SET

for i in range(10):
    FILENAME = 'run{}inter{}'.format(0, i)
    WEIGHTS_PATH = "../data/indexes/recursive_learned_index/" + DATA_SET + FILENAME
    learned_index = RecursiveLearnedIndex([1, 10], [[32, 32], [32, 32]])
    training_data = Model.load_training_data(DATA_PATH + FILENAME)
    learned_index.train(training_data, WEIGHTS_PATH, GRAPH_PATH, EPOCHS)
