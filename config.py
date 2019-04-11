from util.search import SearchType

# N_KEYS:int - Number of keys in the index
N_KEYS = 100000
# MAX_BLOCK_SIZE:int - Max size of blocks where same key appears more than once
MAX_BLOCK_SIZE = 9
# N_INTERPOLATIONS:int - For calculating a spread of how likely a jump is
N_INTERPOLATIONS = 10
# N_RUNS:int - Number of iterations with different seeds (default: 100)
N_RUNS = 1
# N_SAMPLES:int - Number of points where access is sampled
N_SAMPLES = 1000

# NAIVE_COMPLEXITY: list - number of neurons for each layer of the neural net
NAIVE_COMPLEXITY = [32, 32]
# RECURSIVE_SHAPE: list - number of models for each layer of the index tree
RECURSIVE_SHAPE = [1, 10]
# RECURSIVE_COMPLEXITY: list list - complexities of the nets in the index tree
RECURSIVE_COMPLEXITY = [[32, 32], [32, 32]]
# STEP_SIZE:float - the step size used in training for the neural network
STEP_SIZE = 0.001

# SEARCH:enum - Determine which search to use
SEARCH = SearchType.LINEAR

# INDEX:str - Type of index to be used
INDEX = "naive_learned_index"
# INTEGER_DATASET:str - The synthetic dataset of integers
INTEGER_DATASET = "Integers_100x10x100k"
# REAL_WORLD_DATASET:str - The real world dataset of credit card transactions
REAL_WORLD_DATASET = "Creditcard_285k"
# DATASET_PATH:str - Path where datasets are stored
DATASET_PATH = "data/datasets/"
# INTEGER_DATASET_PATH:str - Path where integer dataset is stored
INTEGER_DATASET_PATH = DATASET_PATH + INTEGER_DATASET+"/"
# MODEL_PATH:str - Path where naive_models_0 are stored
MODEL_PATH = "data/indexes/"
# MODEL_PATH:str - Path where prediction results are stored
PREDICTIONS_PATH = "data/predictions/"
# GRAPH_PATH:str - Path where graphs will be stored
GRAPH_PATH = "data/graphs/"
