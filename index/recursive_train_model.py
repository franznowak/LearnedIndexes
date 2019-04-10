from index.recursive_learned_index import RecursiveLearnedIndex
from index.naive_learned_index import Model

EPOCHS = 2000
DATA_SET = "Integers_100x10x100k/"
WEIGHTS_PATH = "../data/indexes/recursive_learned_index/" + DATA_SET
DATA_PATH = "../data/datasets/" + DATA_SET
GRAPH_PATH = "../data/graphs/recursive_learned_index/" + DATA_SET
FILENAME = '{}run{}inter{}'.format(DATA_PATH, 0, 9)

learned_index = RecursiveLearnedIndex([1, 10], [[32, 32], [32, 32]])
training_data = Model.load_training_data(FILENAME)
learned_index.train(training_data, WEIGHTS_PATH, GRAPH_PATH, EPOCHS)
